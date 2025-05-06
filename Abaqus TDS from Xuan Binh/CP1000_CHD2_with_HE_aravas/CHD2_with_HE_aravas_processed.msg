1

   Abaqus 2023.HF4                                  Date 27-Oct-2024   Time 19:02:09
   For use by AALTO UNIVERSITY under license from Dassault Systemes or its subsidiary.

                                                                                 
                                                                                 
 STEP    1     INCREMENT     1     STEP TIME    0.00    


                        S T E P       1     C O U P L E D - T E M P E R A T U R E - D I S P L A C E M E N T   A N A L Y S I S


                                                                                          

          FIXED TIME INCREMENTS
          TIME INCREMENT IS                                     720.    
          TIME PERIOD IS                                       7.200E+03

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

     CHARACTERISTIC ELEMENT LENGTH     1.962E-04

     DETAILS REGARDING ACTUAL SOLUTION WAVEFRONT REQUESTED

     DETAILED OUTPUT OF DIAGNOSTICS TO DATABASE REQUESTED

     PRINT OF INCREMENT NUMBER, TIME, ETC., TO THE MESSAGE FILE EVERY     1  INCREMENTS

     COLLECTING MODEL CONSTRAINT INFORMATION FOR OVERCONSTRAINT CHECKS

     COLLECTING STEP CONSTRAINT INFORMATION FOR OVERCONSTRAINT CHECKS


  INCREMENT     1 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   720.    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  22s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       0.00       TIME AVG. FORCE       1.000E-02
 ALL FORCE     RESIDUALS ARE ZERO
 ALL DISP.    INCREMENTS ARE ZERO
 ALL CORRECTION TO DISP.    ARE ZERO
          THERE IS ZERO FORCE     EVERYWHERE

 AVERAGE HEAT FLUX                  7.998E-14   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         1.206E-18   AT NODE      43484   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.          8.72       AT NODE          1   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.         8.72       AT NODE          1   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   720.    ,  FRACTION OF STEP COMPLETED  0.100    
 STEP TIME COMPLETED        720.    ,  TOTAL TIME COMPLETED         720.    


  INCREMENT     2 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   720.    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  24s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       0.00       TIME AVG. FORCE       1.000E-02
 ALL FORCE     RESIDUALS ARE ZERO
 ALL DISP.    INCREMENTS ARE ZERO
 ALL CORRECTION TO DISP.    ARE ZERO
          THERE IS ZERO FORCE     EVERYWHERE

 AVERAGE HEAT FLUX                  8.383E-16   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         1.736E-24   AT NODE      43157   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         0.143       AT NODE      37147   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        -8.72       AT NODE          1   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   720.    ,  FRACTION OF STEP COMPLETED  0.200    
 STEP TIME COMPLETED       1.440E+03,  TOTAL TIME COMPLETED        1.440E+03


  INCREMENT     3 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   720.    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  21s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       0.00       TIME AVG. FORCE       1.000E-02
 ALL FORCE     RESIDUALS ARE ZERO
 ALL DISP.    INCREMENTS ARE ZERO
 ALL CORRECTION TO DISP.    ARE ZERO
          THERE IS ZERO FORCE     EVERYWHERE

 AVERAGE HEAT FLUX                  1.016E-17   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -2.903E-25   AT NODE      42963   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         2.034E-03   AT NODE      37147   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -0.141       AT NODE      37147   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   720.    ,  FRACTION OF STEP COMPLETED  0.300    
 STEP TIME COMPLETED       2.160E+03,  TOTAL TIME COMPLETED        2.160E+03


  INCREMENT     4 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   720.    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  26s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       0.00       TIME AVG. FORCE       1.000E-02
 ALL FORCE     RESIDUALS ARE ZERO
 ALL DISP.    INCREMENTS ARE ZERO
 ALL CORRECTION TO DISP.    ARE ZERO
          THERE IS ZERO FORCE     EVERYWHERE

 AVERAGE HEAT FLUX                  1.248E-19   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -4.403E-25   AT NODE      40819   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         2.962E-05   AT NODE      39854   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.006E-03   AT NODE      37147   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   720.    ,  FRACTION OF STEP COMPLETED  0.400    
 STEP TIME COMPLETED       2.880E+03,  TOTAL TIME COMPLETED        2.880E+03


  INCREMENT     5 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   720.    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  22s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       0.00       TIME AVG. FORCE       1.000E-02
 ALL FORCE     RESIDUALS ARE ZERO
 ALL DISP.    INCREMENTS ARE ZERO
 ALL CORRECTION TO DISP.    ARE ZERO
          THERE IS ZERO FORCE     EVERYWHERE

 AVERAGE HEAT FLUX                  1.575E-21   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -4.589E-25   AT NODE      42745   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         4.543E-07   AT NODE      39829   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.917E-05   AT NODE      39854   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   720.    ,  FRACTION OF STEP COMPLETED  0.500    
 STEP TIME COMPLETED       3.600E+03,  TOTAL TIME COMPLETED        3.600E+03


  INCREMENT     6 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   720.    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  21s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       0.00       TIME AVG. FORCE       1.000E-02
 ALL FORCE     RESIDUALS ARE ZERO
 ALL DISP.    INCREMENTS ARE ZERO
 ALL CORRECTION TO DISP.    ARE ZERO
          THERE IS ZERO FORCE     EVERYWHERE

 AVERAGE HEAT FLUX                  2.224E-23   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -4.061E-25   AT NODE      42025   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         7.335E-09   AT NODE      40057   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -4.470E-07   AT NODE      39829   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   720.    ,  FRACTION OF STEP COMPLETED  0.600    
 STEP TIME COMPLETED       4.320E+03,  TOTAL TIME COMPLETED        4.320E+03


  INCREMENT     7 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   720.    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  21s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       0.00       TIME AVG. FORCE       1.000E-02
 ALL FORCE     RESIDUALS ARE ZERO
 ALL DISP.    INCREMENTS ARE ZERO
 ALL CORRECTION TO DISP.    ARE ZERO
          THERE IS ZERO FORCE     EVERYWHERE

 AVERAGE HEAT FLUX                  3.763E-25   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         4.509E-25   AT NODE      41991   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         1.271E-10   AT NODE      41667   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -7.209E-09   AT NODE      40057   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   720.    ,  FRACTION OF STEP COMPLETED  0.700    
 STEP TIME COMPLETED       5.040E+03,  TOTAL TIME COMPLETED        5.040E+03


  INCREMENT     8 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   720.    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  21s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       0.00       TIME AVG. FORCE       1.000E-02
 ALL FORCE     RESIDUALS ARE ZERO
 ALL DISP.    INCREMENTS ARE ZERO
 ALL CORRECTION TO DISP.    ARE ZERO
          THERE IS ZERO FORCE     EVERYWHERE

 AVERAGE HEAT FLUX                  7.575E-27   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         4.531E-25   AT NODE      42752   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         2.386E-12   AT NODE      41667   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.247E-10   AT NODE      41667   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   720.    ,  FRACTION OF STEP COMPLETED  0.800    
 STEP TIME COMPLETED       5.760E+03,  TOTAL TIME COMPLETED        5.760E+03


  INCREMENT     9 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   720.    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  21s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       0.00       TIME AVG. FORCE       1.000E-02
 ALL FORCE     RESIDUALS ARE ZERO
 ALL DISP.    INCREMENTS ARE ZERO
 ALL CORRECTION TO DISP.    ARE ZERO
          THERE IS ZERO FORCE     EVERYWHERE

 AVERAGE HEAT FLUX                  7.538E-28   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.047E-25   AT NODE      43470   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         5.034E-14   AT NODE      41667   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.335E-12   AT NODE      41667   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   720.    ,  FRACTION OF STEP COMPLETED  0.900    
 STEP TIME COMPLETED       6.480E+03,  TOTAL TIME COMPLETED        6.480E+03


  INCREMENT    10 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   720.    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  21s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       0.00       TIME AVG. FORCE       1.000E-02
 ALL FORCE     RESIDUALS ARE ZERO
 ALL DISP.    INCREMENTS ARE ZERO
 ALL CORRECTION TO DISP.    ARE ZERO
          THERE IS ZERO FORCE     EVERYWHERE

 AVERAGE HEAT FLUX                  5.515E-28   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -4.209E-25   AT NODE      40947   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         3.372E-15   AT NODE      38174   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        4.977E-14   AT NODE      41701   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   720.    ,  FRACTION OF STEP COMPLETED   1.00    
 STEP TIME COMPLETED       7.200E+03,  TOTAL TIME COMPLETED        7.200E+03
1

   Abaqus 2023.HF4                                  Date 27-Oct-2024   Time 19:06:56
   For use by AALTO UNIVERSITY under license from Dassault Systemes or its subsidiary.

                                                                                 
                                                                                 
 STEP    2     INCREMENT     1     STEP TIME    0.00    


                        S T E P       2     C O U P L E D - T E M P E R A T U R E - D I S P L A C E M E N T   A N A L Y S I S


                                                                                          

     AUTOMATIC TIME CONTROL WITH -
          A SUGGESTED INITIAL TIME INCREMENT OF                 6.65    
          AND A TOTAL TIME PERIOD OF                            665.    
          THE MINIMUM TIME INCREMENT ALLOWED IS                1.000E-07
          THE MAXIMUM TIME INCREMENT ALLOWED IS                 6.65    

          CREEP AND SWELLING EFFECTS ARE OMITTED IN THIS STEP

     THE SIZE OF THE TIME INCREMENT IS CONTROLLED BY -
          THE TEMPERATURE CHANGE PER INCREMENT NOT EXCEEDING            10.0    
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

     CHARACTERISTIC ELEMENT LENGTH     1.962E-04

     DETAILS REGARDING ACTUAL SOLUTION WAVEFRONT REQUESTED

     DETAILED OUTPUT OF DIAGNOSTICS TO DATABASE REQUESTED

     PRINT OF INCREMENT NUMBER, TIME, ETC., TO THE MESSAGE FILE EVERY     1  INCREMENTS

     COLLECTING STEP CONSTRAINT INFORMATION FOR OVERCONSTRAINT CHECKS


  INCREMENT     1 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  4.9s

 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 4775 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 4774 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 10653 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 30300 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 9081 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 4777 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 3205 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 24968 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 27634 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 6143 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 7715 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 27640 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 30306 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 4776 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 3204 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 24974 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 30318 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 27652 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 30312 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 3202 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 3203 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 24980 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 24986 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 27646 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 10648 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 7710 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 30330 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 27664 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 9076 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 6138 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 16524 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 20820 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 14952 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 16525 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 20808 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 20814 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 14953 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 16529 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 20790 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 14957 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 20796 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 16527 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 16528 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 14956 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 20802 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 14954 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 14955 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 16526 INSTANCE 
             CHD2 D.O.F. 11
  Typically a zero pivot is associated with an overconstraint in the model. 
 However, Abaqus was not able to identify any chains of constraints that might 
 lead to an overconstraint at this node. This suggests that the model might be 
 in fact insufficiently constrained in this particular degree of freedom. 
 Please check the constraints used in this model. 


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING NODE 32996 INSTANCE 
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


 AVERAGE FORCE                      6.543E-02   TIME AVG. FORCE       6.543E-02
 LARGEST SCALED RESIDUAL FORCE      1.610E-03   AT NODE      43088   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      1.610E-03
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.329E-16   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -6.632E-14   AT NODE      42810   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         2.187E-15   AT NODE      37567   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.187E-15   AT NODE      37567   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  4.9s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      6.543E-02   TIME AVG. FORCE       6.543E-02
 LARGEST SCALED RESIDUAL FORCE     -2.910E-05   AT NODE      41655   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -2.910E-05
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        8.108E-11   AT NODE      42139   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.391E-16   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         3.526E-18   AT NODE      42810   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.461E-04   AT NODE      42526   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -3.461E-04   AT NODE      42526   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -3.461E-04

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  1.000E-02
 STEP TIME COMPLETED        6.65    ,  TOTAL TIME COMPLETED        7.207E+03


  INCREMENT     2 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  4.9s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.165       TIME AVG. FORCE       0.115    
 LARGEST SCALED RESIDUAL FORCE      2.269E-05   AT NODE      40899   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      2.269E-05
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.621E-10   AT NODE      42139   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  3.278E-16   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         7.051E-18   AT NODE      42810   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         6.137E-03   AT NODE      20897   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        6.299E-03   AT NODE      20897   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   6.137E-03
          TIME INCREMENT MAY NOW INCREASE TO     6.65    

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  2.000E-02
 STEP TIME COMPLETED        13.3    ,  TOTAL TIME COMPLETED        7.213E+03


  INCREMENT     3 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  4.9s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.248       TIME AVG. FORCE       0.159    
 LARGEST SCALED RESIDUAL FORCE     -1.916E-05   AT NODE      40899   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -1.916E-05
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.621E-10   AT NODE      42139   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  5.046E-16   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         7.051E-18   AT NODE      42810   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         7.000E-03   AT NODE      20897   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        9.849E-04   AT NODE      14864   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   7.000E-03

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  3.000E-02
 STEP TIME COMPLETED        20.0    ,  TOTAL TIME COMPLETED        7.220E+03


  INCREMENT     4 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  4.9s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.330       TIME AVG. FORCE       0.202    
 LARGEST SCALED RESIDUAL FORCE      2.142E-05   AT NODE      41024   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      2.142E-05
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.621E-10   AT NODE      42139   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  6.730E-16   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         7.050E-18   AT NODE      42810   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         7.177E-03   AT NODE      20897   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.778E-04   AT NODE      14863   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   7.177E-03

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  4.000E-02
 STEP TIME COMPLETED        26.6    ,  TOTAL TIME COMPLETED        7.227E+03


  INCREMENT     5 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  5.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.413       TIME AVG. FORCE       0.244    
 LARGEST SCALED RESIDUAL FORCE     -2.130E-05   AT NODE      40620   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -2.130E-05
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.621E-10   AT NODE      42139   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  8.370E-16   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         7.050E-18   AT NODE      42810   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         7.225E-03   AT NODE      20897   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.647E-04   AT NODE      43050   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   7.225E-03

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  5.000E-02
 STEP TIME COMPLETED        33.2    ,  TOTAL TIME COMPLETED        7.233E+03


  INCREMENT     6 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  5.0s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.495       TIME AVG. FORCE       0.286    
 LARGEST SCALED RESIDUAL FORCE      4.584E-03   AT NODE       9064   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      4.584E-03
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -7.352E-09   AT NODE          5   DOF  3
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  9.983E-16   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.304E-17   AT NODE      40782   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         7.239E-03   AT NODE      20897   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.126E-04   AT NODE      43050   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  4.9s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.495       TIME AVG. FORCE       0.286    
 LARGEST SCALED RESIDUAL FORCE      2.967E-04   AT NODE       9064   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      2.967E-04
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -2.585E-10   AT NODE      11762   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  9.983E-16   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         3.248E-19   AT NODE      12160   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         7.238E-03   AT NODE      20897   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -5.144E-06   AT NODE      14944   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   7.238E-03

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  6.000E-02
 STEP TIME COMPLETED        39.9    ,  TOTAL TIME COMPLETED        7.240E+03


  INCREMENT     7 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  5.0s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.577       TIME AVG. FORCE       0.328    
 LARGEST SCALED RESIDUAL FORCE     -3.260E-02   AT NODE      30407   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -3.260E-02
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -3.180E-08   AT NODE          5   DOF  3
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.156E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         7.515E-17   AT NODE      40918   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         1.240E-02   AT NODE      14943   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        6.758E-03   AT NODE      14944   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  5.0s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.577       TIME AVG. FORCE       0.328    
 LARGEST SCALED RESIDUAL FORCE     -1.157E-03   AT NODE      27742   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -1.157E-03
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -3.795E-09   AT NODE          6   DOF  3
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.156E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         6.861E-18   AT NODE      27742   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         1.239E-02   AT NODE      14943   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.691E-05   AT NODE      14863   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   1.239E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  7.000E-02
 STEP TIME COMPLETED        46.5    ,  TOTAL TIME COMPLETED        7.247E+03


  INCREMENT     8 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  4.9s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.658       TIME AVG. FORCE       0.369    
 LARGEST SCALED RESIDUAL FORCE     -3.775E-02   AT NODE      27929   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -3.775E-02
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -3.193E-08   AT NODE        253   DOF  3
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.301E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         1.195E-16   AT NODE      41968   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         4.941E-02   AT NODE      14944   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        3.791E-02   AT NODE      14864   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  5.0s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.658       TIME AVG. FORCE       0.369    
 LARGEST SCALED RESIDUAL FORCE     -9.689E-04   AT NODE      25035   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -9.689E-04
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -3.498E-09   AT NODE      27929   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.301E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         6.163E-18   AT NODE      40884   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         4.941E-02   AT NODE      14944   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.773E-05   AT NODE      20869   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   4.941E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  8.000E-02
 STEP TIME COMPLETED        53.2    ,  TOTAL TIME COMPLETED        7.253E+03


  INCREMENT     9 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  4.9s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.738       TIME AVG. FORCE       0.410    
 LARGEST SCALED RESIDUAL FORCE      2.863E-02   AT NODE       6049   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      2.863E-02
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -3.737E-08   AT NODE        254   DOF  3
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.449E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         1.563E-16   AT NODE      40918   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         6.203E-02   AT NODE      14945   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        3.371E-02   AT NODE      14863   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  4.9s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.738       TIME AVG. FORCE       0.410    
 LARGEST SCALED RESIDUAL FORCE     -1.753E-03   AT NODE      25264   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -1.753E-03
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -2.757E-09   AT NODE        173   DOF  3
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.449E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         7.325E-18   AT NODE      41968   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         6.204E-02   AT NODE      14945   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        3.872E-05   AT NODE      21127   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   6.204E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  9.000E-02
 STEP TIME COMPLETED        59.8    ,  TOTAL TIME COMPLETED        7.260E+03


  INCREMENT    10 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  5.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.816       TIME AVG. FORCE       0.451    
 LARGEST SCALED RESIDUAL FORCE     -3.339E-02   AT NODE      33278   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -3.339E-02
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -4.483E-08   AT NODE        173   DOF  3
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.715E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         2.016E-16   AT NODE      41968   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         6.815E-02   AT NODE      14946   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        3.261E-02   AT NODE      14862   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  5.0s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.816       TIME AVG. FORCE       0.451    
 LARGEST SCALED RESIDUAL FORCE     -4.315E-03   AT NODE      33264   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -4.315E-03
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -4.837E-09   AT NODE        256   DOF  3
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.715E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         1.320E-17   AT NODE      40918   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         6.817E-02   AT NODE      14946   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        5.236E-05   AT NODE      16729   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  5.0s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      0.816       TIME AVG. FORCE       0.451    
 LARGEST SCALED RESIDUAL FORCE     -2.712E-05   AT NODE      41912   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -2.712E-05
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -2.734E-10   AT NODE      33264   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.715E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         4.147E-19   AT NODE      33278   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         6.817E-02   AT NODE      14946   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        5.715E-06   AT NODE      21115   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   6.817E-02

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.100    
 STEP TIME COMPLETED        66.5    ,  TOTAL TIME COMPLETED        7.266E+03


  INCREMENT    11 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  5.3s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.892       TIME AVG. FORCE       0.491    
 LARGEST SCALED RESIDUAL FORCE      3.460E-02   AT NODE       8984   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      3.460E-02
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -5.341E-08   AT NODE      18733   DOF  3
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  1.898E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         2.979E-16   AT NODE      40884   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         6.810E-02   AT NODE      14947   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -3.173E-02   AT NODE       3193   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  5.3s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.892       TIME AVG. FORCE       0.491    
 LARGEST SCALED RESIDUAL FORCE     -2.389E-03   AT NODE      21090   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -2.389E-03
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        5.806E-09   AT NODE      16627   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  1.898E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         2.166E-17   AT NODE      41968   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         6.815E-02   AT NODE      14947   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        7.386E-05   AT NODE       8986   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   6.815E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.110    
 STEP TIME COMPLETED        73.2    ,  TOTAL TIME COMPLETED        7.273E+03


  INCREMENT    12 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  5.2s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.964       TIME AVG. FORCE       0.530    
 LARGEST SCALED RESIDUAL FORCE     -6.759E-02   AT NODE      25477   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -6.759E-02
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        8.016E-08   AT NODE       5802   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  2.091E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         5.648E-16   AT NODE      41934   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         5.884E-02   AT NODE      14948   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -3.084E-02   AT NODE      20912   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  5.3s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.963       TIME AVG. FORCE       0.530    
 LARGEST SCALED RESIDUAL FORCE      6.425E-03   AT NODE      12014   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      6.425E-03
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        7.825E-09   AT NODE      13330   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  2.091E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         4.620E-17   AT NODE      41934   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         5.898E-02   AT NODE      14948   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.402E-04   AT NODE      14948   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  5.0s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      0.963       TIME AVG. FORCE       0.530    
 LARGEST SCALED RESIDUAL FORCE     -3.706E-05   AT NODE      13323   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -3.706E-05
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        3.555E-10   AT NODE      16263   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  2.091E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         6.333E-19   AT NODE      41934   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         5.898E-02   AT NODE      14948   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.269E-05   AT NODE      16267   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   5.898E-02

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.120    
 STEP TIME COMPLETED        79.8    ,  TOTAL TIME COMPLETED        7.280E+03


  INCREMENT    13 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  5.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.02       TIME AVG. FORCE       0.568    
 LARGEST SCALED RESIDUAL FORCE     -0.122       AT NODE      32605   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -0.122    
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.826E-07   AT NODE      15696   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  2.293E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         2.166E-15   AT NODE      40918   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.430E-02   AT NODE      16301   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        4.073E-02   AT NODE      20847   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  5.3s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.01       TIME AVG. FORCE       0.567    
 LARGEST SCALED RESIDUAL FORCE      1.205E-02   AT NODE       3692   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      1.205E-02
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.899E-08   AT NODE      14459   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  2.292E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         2.642E-16   AT NODE      40884   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.432E-02   AT NODE      16301   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.524E-04   AT NODE      14951   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  8.5s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.01       TIME AVG. FORCE       0.567    
 LARGEST SCALED RESIDUAL FORCE      1.266E-03   AT NODE      13443   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      1.266E-03
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.430E-09   AT NODE       8538   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  2.292E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         1.366E-17   AT NODE      40918   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.433E-02   AT NODE      16301   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.284E-05   AT NODE      20092   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -4.433E-02

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.130    
 STEP TIME COMPLETED        86.5    ,  TOTAL TIME COMPLETED        7.286E+03


  INCREMENT    14 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.7s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.993       TIME AVG. FORCE       0.598    
 LARGEST SCALED RESIDUAL FORCE      0.546       AT NODE      39715   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      0.546    
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        5.036E-07   AT NODE      17402   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  2.518E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         7.499E-15   AT NODE      41968   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.470E-02   AT NODE      20867   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        5.752E-02   AT NODE      16784   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  5.0s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.990       TIME AVG. FORCE       0.598    
 LARGEST SCALED RESIDUAL FORCE     -0.454       AT NODE      37068   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -0.454    
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.514E-07   AT NODE      37531   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  2.511E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         1.310E-15   AT NODE      40952   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.468E-02   AT NODE      20867   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        3.540E-04   AT NODE      14955   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  5.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      0.984       TIME AVG. FORCE       0.597    
 LARGEST SCALED RESIDUAL FORCE     -0.270       AT NODE      36980   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -0.270    
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.346E-07   AT NODE      37688   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  2.508E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         8.834E-16   AT NODE      42733   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.469E-02   AT NODE      20867   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.002E-04   AT NODE      21862   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  5.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      0.982       TIME AVG. FORCE       0.597    
 LARGEST SCALED RESIDUAL FORCE     -6.644E-02   AT NODE      34976   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -6.644E-02
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -7.125E-08   AT NODE        264   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  2.506E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         3.316E-16   AT NODE      40782   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.469E-02   AT NODE      20867   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        4.663E-05   AT NODE      14878   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  5.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      0.982       TIME AVG. FORCE       0.597    
 LARGEST SCALED RESIDUAL FORCE     -3.833E-02   AT NODE      34976   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -3.833E-02
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        5.188E-09   AT NODE      37974   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  2.506E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -4.563E-17   AT NODE      36973   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.469E-02   AT NODE      20867   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.951E-05   AT NODE      37226   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  5.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     6


 AVERAGE FORCE                      0.982       TIME AVG. FORCE       0.597    
 LARGEST SCALED RESIDUAL FORCE     -7.847E-04   AT NODE      36969   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -7.847E-04
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -2.030E-09   AT NODE      37689   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  2.506E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -6.245E-18   AT NODE      39719   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.469E-02   AT NODE      20867   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -4.451E-06   AT NODE      35577   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -3.469E-02

 ITERATION SUMMARY FOR THE INCREMENT:   6 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  6 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.140    
 STEP TIME COMPLETED        93.1    ,  TOTAL TIME COMPLETED        7.293E+03


  INCREMENT    15 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  5.0s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.965       TIME AVG. FORCE       0.622    
 LARGEST SCALED RESIDUAL FORCE      -9.16       AT NODE      40410   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      -9.16    
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        5.494E-06   AT NODE      40343   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  2.874E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         4.032E-14   AT NODE      42733   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.396E-02   AT NODE      14953   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -4.357E-02   AT NODE      14855   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  5.0s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.990       TIME AVG. FORCE       0.623    
 LARGEST SCALED RESIDUAL FORCE      0.345       AT NODE      39963   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      0.345    
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        6.317E-07   AT NODE      37532   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  2.813E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         6.306E-15   AT NODE      41018   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.350E-02   AT NODE      14953   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -7.903E-04   AT NODE      14944   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  4.9s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      0.990       TIME AVG. FORCE       0.623    
 LARGEST SCALED RESIDUAL FORCE     -2.054E-02   AT NODE      36854   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -2.054E-02
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        6.930E-08   AT NODE      16794   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  2.807E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         3.093E-16   AT NODE      42733   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.386E-02   AT NODE      14953   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -3.915E-04   AT NODE      14854   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  4.9s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      0.990       TIME AVG. FORCE       0.623    
 LARGEST SCALED RESIDUAL FORCE     -3.004E-03   AT NODE      34884   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -3.004E-03
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -5.395E-09   AT NODE      35453   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  2.807E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         1.413E-17   AT NODE      42002   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.388E-02   AT NODE      14953   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -4.277E-05   AT NODE      14958   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -4.388E-02

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.150    
 STEP TIME COMPLETED        99.8    ,  TOTAL TIME COMPLETED        7.300E+03


  INCREMENT    16 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  5.0s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.980       TIME AVG. FORCE       0.645    
 LARGEST SCALED RESIDUAL FORCE     -0.325       AT NODE      37094   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -0.325    
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        9.022E-07   AT NODE      40058   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.396E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         8.224E-15   AT NODE      40781   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.101       AT NODE      14850   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -9.454E-02   AT NODE      14849   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  5.0s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.00       TIME AVG. FORCE       0.647    
 LARGEST SCALED RESIDUAL FORCE      8.496E-02   AT NODE      22409   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      8.496E-02
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.179E-07   AT NODE      37239   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.380E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         6.528E-16   AT NODE      41934   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.101       AT NODE      14850   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -8.108E-04   AT NODE      14864   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  5.0s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.00       TIME AVG. FORCE       0.647    
 LARGEST SCALED RESIDUAL FORCE      1.446E-02   AT NODE      34280   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      1.446E-02
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        4.105E-08   AT NODE      37695   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.379E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         8.423E-17   AT NODE      41968   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.101       AT NODE      14850   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.087E-04   AT NODE      14963   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  5.0s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                       1.00       TIME AVG. FORCE       0.647    
 LARGEST SCALED RESIDUAL FORCE     -3.498E-03   AT NODE      34279   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -3.498E-03
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -4.001E-09   AT NODE      21790   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.379E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         8.435E-18   AT NODE      41934   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.101       AT NODE      14850   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.753E-05   AT NODE      14966   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  5.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                       1.00       TIME AVG. FORCE       0.647    
 LARGEST SCALED RESIDUAL FORCE     -1.863E-04   AT NODE      10248   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -1.863E-04
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        4.493E-10   AT NODE      16162   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  3.379E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         6.930E-19   AT NODE      40884   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -0.101       AT NODE      14850   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.878E-06   AT NODE      14968   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -0.101    

 ITERATION SUMMARY FOR THE INCREMENT:   5 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  5 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.160    
 STEP TIME COMPLETED        106.    ,  TOTAL TIME COMPLETED        7.306E+03


  INCREMENT    17 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  5.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.00       TIME AVG. FORCE       0.668    
 LARGEST SCALED RESIDUAL FORCE     -0.114       AT NODE      37520   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -0.114    
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.716E-07   AT NODE      37308   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.297E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         2.570E-15   AT NODE      40986   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -7.082E-02   AT NODE      17562   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        8.798E-02   AT NODE      14851   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  5.0s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.01       TIME AVG. FORCE       0.668    
 LARGEST SCALED RESIDUAL FORCE     -3.620E-02   AT NODE      12077   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -3.620E-02
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        7.510E-08   AT NODE      14866   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.292E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         2.302E-16   AT NODE      42002   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -7.087E-02   AT NODE      17562   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -7.672E-04   AT NODE      14864   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  5.0s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.01       TIME AVG. FORCE       0.668    
 LARGEST SCALED RESIDUAL FORCE     -6.066E-03   AT NODE      33578   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -6.066E-03
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.170E-08   AT NODE      15017   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.292E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         2.373E-17   AT NODE      40952   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -7.088E-02   AT NODE      17562   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -3.043E-05   AT NODE      14838   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  5.0s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                       1.00       TIME AVG. FORCE       0.668    
 LARGEST SCALED RESIDUAL FORCE      7.664E-04   AT NODE      33599   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      7.664E-04
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.190E-09   AT NODE      16112   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  3.292E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         1.535E-18   AT NODE      41968   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -7.088E-02   AT NODE      17562   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -5.182E-06   AT NODE      14835   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -7.088E-02

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.170    
 STEP TIME COMPLETED        113.    ,  TOTAL TIME COMPLETED        7.313E+03


  INCREMENT    18 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  5.3s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.01       TIME AVG. FORCE       0.687    
 LARGEST SCALED RESIDUAL FORCE      0.115       AT NODE      39860   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      0.115    
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.922E-07   AT NODE      38063   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.205E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         1.869E-15   AT NODE      40816   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         6.514E-02   AT NODE      20120   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        5.287E-02   AT NODE      14845   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  7.7s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.01       TIME AVG. FORCE       0.687    
 LARGEST SCALED RESIDUAL FORCE      1.380E-02   AT NODE      22415   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      1.380E-02
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        3.454E-08   AT NODE      15376   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.203E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         1.628E-16   AT NODE      40850   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         6.515E-02   AT NODE      20120   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -7.081E-04   AT NODE      14864   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  8.4s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.01       TIME AVG. FORCE       0.687    
 LARGEST SCALED RESIDUAL FORCE      2.454E-03   AT NODE      11858   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      2.454E-03
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -3.857E-09   AT NODE      23277   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  3.202E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         1.079E-17   AT NODE      40884   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         6.515E-02   AT NODE      20120   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.082E-05   AT NODE      14832   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   6.515E-02

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.180    
 STEP TIME COMPLETED        120.    ,  TOTAL TIME COMPLETED        7.320E+03


  INCREMENT    19 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  8.2s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.02       TIME AVG. FORCE       0.704    
 LARGEST SCALED RESIDUAL FORCE      5.493E-02   AT NODE      39955   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      5.493E-02
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.658E-07   AT NODE          5   DOF  3
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.153E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         1.099E-15   AT NODE      39907   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         6.182E-02   AT NODE      21076   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.855E-02   AT NODE      15191   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  8.2s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.02       TIME AVG. FORCE       0.704    
 LARGEST SCALED RESIDUAL FORCE      8.210E-03   AT NODE      14614   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      8.210E-03
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.776E-08   AT NODE      14795   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.152E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         9.401E-17   AT NODE      39835   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         6.182E-02   AT NODE      21076   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -6.602E-04   AT NODE      14864   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  8.2s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.02       TIME AVG. FORCE       0.704    
 LARGEST SCALED RESIDUAL FORCE      1.038E-03   AT NODE      13726   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      1.038E-03
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.419E-09   AT NODE      16070   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  3.152E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         5.534E-18   AT NODE      40918   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         6.182E-02   AT NODE      21076   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        5.632E-06   AT NODE      35602   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   6.182E-02

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.190    
 STEP TIME COMPLETED        126.    ,  TOTAL TIME COMPLETED        7.326E+03


  INCREMENT    20 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.3s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.02       TIME AVG. FORCE       0.720    
 LARGEST SCALED RESIDUAL FORCE     -0.159       AT NODE      41020   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -0.159    
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.465E-07   AT NODE          5   DOF  3
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.137E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         1.032E-15   AT NODE      40986   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         7.406E-02   AT NODE      21077   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.518E-02   AT NODE      21068   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.4s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.02       TIME AVG. FORCE       0.720    
 LARGEST SCALED RESIDUAL FORCE      6.795E-03   AT NODE      14629   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      6.795E-03
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -1.161E-08   AT NODE      24041   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.136E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         6.676E-17   AT NODE      41900   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         7.408E-02   AT NODE      21077   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -6.236E-04   AT NODE      14864   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  8.2s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.02       TIME AVG. FORCE       0.720    
 LARGEST SCALED RESIDUAL FORCE     -6.638E-04   AT NODE      34674   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -6.638E-04
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.142E-09   AT NODE      16052   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  3.136E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         2.528E-18   AT NODE      40918   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         7.408E-02   AT NODE      21077   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -4.597E-06   AT NODE      14825   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   7.408E-02

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.200    
 STEP TIME COMPLETED        133.    ,  TOTAL TIME COMPLETED        7.333E+03


  INCREMENT    21 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.2s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.03       TIME AVG. FORCE       0.735    
 LARGEST SCALED RESIDUAL FORCE     -8.909E-02   AT NODE      37054   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -8.909E-02
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.219E-07   AT NODE          6   DOF  3
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.145E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         8.123E-16   AT NODE      42002   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         6.412E-02   AT NODE      21077   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.202E-02   AT NODE      29568   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.7s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.03       TIME AVG. FORCE       0.735    
 LARGEST SCALED RESIDUAL FORCE      4.424E-03   AT NODE      36082   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      4.424E-03
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        7.014E-09   AT NODE      14695   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.144E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         5.457E-17   AT NODE      41899   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         6.413E-02   AT NODE      21077   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -5.970E-04   AT NODE      14864   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.5s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.03       TIME AVG. FORCE       0.735    
 LARGEST SCALED RESIDUAL FORCE     -6.729E-04   AT NODE      34719   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -6.729E-04
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        7.657E-10   AT NODE      16614   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  3.144E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         1.814E-18   AT NODE      41968   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         6.413E-02   AT NODE      21077   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -3.662E-06   AT NODE      14822   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   6.413E-02

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.210    
 STEP TIME COMPLETED        140.    ,  TOTAL TIME COMPLETED        7.340E+03


  INCREMENT    22 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  8.9s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.04       TIME AVG. FORCE       0.748    
 LARGEST SCALED RESIDUAL FORCE      1.711E-02   AT NODE      14658   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      1.711E-02
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.042E-07   AT NODE      38063   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.169E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         6.398E-16   AT NODE      42002   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         5.767E-02   AT NODE      15191   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.778E-02   AT NODE      21067   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  8.2s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.04       TIME AVG. FORCE       0.748    
 LARGEST SCALED RESIDUAL FORCE     -3.839E-03   AT NODE      33908   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -3.839E-03
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -5.213E-09   AT NODE      14696   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.169E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         2.101E-17   AT NODE      41968   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         5.769E-02   AT NODE      15191   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -5.795E-04   AT NODE      14944   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  8.6s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.04       TIME AVG. FORCE       0.748    
 LARGEST SCALED RESIDUAL FORCE      3.026E-04   AT NODE      33952   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      3.026E-04
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        5.794E-10   AT NODE      16617   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  3.169E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         8.885E-19   AT NODE      41968   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         5.769E-02   AT NODE      15191   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.909E-06   AT NODE      14989   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   5.769E-02

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.220    
 STEP TIME COMPLETED        146.    ,  TOTAL TIME COMPLETED        7.346E+03


  INCREMENT    23 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  8.3s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.04       TIME AVG. FORCE       0.761    
 LARGEST SCALED RESIDUAL FORCE     -1.644E-02   AT NODE      32242   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -1.644E-02
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        9.539E-08   AT NODE      37240   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.203E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         5.214E-16   AT NODE      41830   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         5.662E-02   AT NODE      21068   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.468E-02   AT NODE      20203   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  8.3s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.04       TIME AVG. FORCE       0.761    
 LARGEST SCALED RESIDUAL FORCE     -1.253E-03   AT NODE      33974   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -1.253E-03
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        4.568E-09   AT NODE      14695   DOF  1
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  3.202E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         1.253E-17   AT NODE      41968   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         5.662E-02   AT NODE      21068   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -5.695E-04   AT NODE      14944   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   5.662E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.230    
 STEP TIME COMPLETED        153.    ,  TOTAL TIME COMPLETED        7.353E+03


  INCREMENT    24 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.4s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.05       TIME AVG. FORCE       0.773    
 LARGEST SCALED RESIDUAL FORCE     -1.106E-02   AT NODE      32418   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -1.106E-02
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        9.041E-08   AT NODE      37693   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.239E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.036E-16   AT NODE      11927   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         4.923E-02   AT NODE      21068   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.368E-02   AT NODE      21077   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.5s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.05       TIME AVG. FORCE       0.773    
 LARGEST SCALED RESIDUAL FORCE      6.875E-04   AT NODE      12061   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      6.875E-04
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        3.775E-09   AT NODE      14695   DOF  1
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  3.239E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         6.089E-18   AT NODE      42002   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         4.924E-02   AT NODE      21068   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -5.644E-04   AT NODE      14945   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   4.924E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.240    
 STEP TIME COMPLETED        160.    ,  TOTAL TIME COMPLETED        7.360E+03


  INCREMENT    25 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  8.8s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.06       TIME AVG. FORCE       0.785    
 LARGEST SCALED RESIDUAL FORCE      8.559E-03   AT NODE      14674   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      8.559E-03
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        8.665E-08   AT NODE      38064   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.276E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -4.949E-16   AT NODE      11927   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         4.498E-02   AT NODE      21069   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.303E-02   AT NODE      16509   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  8.7s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.06       TIME AVG. FORCE       0.785    
 LARGEST SCALED RESIDUAL FORCE     -2.588E-04   AT NODE      27053   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -2.588E-04
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        3.142E-09   AT NODE      14695   DOF  1
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  3.275E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -4.240E-18   AT NODE      41832   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         4.499E-02   AT NODE      21069   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -5.669E-04   AT NODE      14945   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   4.499E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.250    
 STEP TIME COMPLETED        166.    ,  TOTAL TIME COMPLETED        7.366E+03


  INCREMENT    26 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  8.7s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.06       TIME AVG. FORCE       0.795    
 LARGEST SCALED RESIDUAL FORCE     -1.181E-02   AT NODE      26900   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -1.181E-02
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        8.405E-08   AT NODE        253   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.312E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -4.914E-16   AT NODE      11927   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         4.608E-02   AT NODE      21069   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.124E-02   AT NODE      30588   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  8.8s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.06       TIME AVG. FORCE       0.795    
 LARGEST SCALED RESIDUAL FORCE     -3.326E-04   AT NODE      38952   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -3.326E-04
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        3.221E-09   AT NODE      20861   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  3.312E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -4.548E-18   AT NODE      41832   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         4.608E-02   AT NODE      21069   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -5.731E-04   AT NODE      14945   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   4.608E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.260    
 STEP TIME COMPLETED        173.    ,  TOTAL TIME COMPLETED        7.373E+03


  INCREMENT    27 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  8.7s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.07       TIME AVG. FORCE       0.805    
 LARGEST SCALED RESIDUAL FORCE     -4.977E-03   AT NODE      29737   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -4.977E-03
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -9.056E-08   AT NODE        175   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.347E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -4.904E-16   AT NODE      11927   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         4.021E-02   AT NODE      21069   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -9.913E-03   AT NODE      21068   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  8.9s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.07       TIME AVG. FORCE       0.805    
 LARGEST SCALED RESIDUAL FORCE      1.079E-04   AT NODE      33141   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      1.079E-04
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -3.147E-09   AT NODE      16622   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  3.346E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -3.574E-18   AT NODE      42070   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         4.022E-02   AT NODE      21069   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -5.814E-04   AT NODE      14945   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   4.022E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.270    
 STEP TIME COMPLETED        180.    ,  TOTAL TIME COMPLETED        7.380E+03


  INCREMENT    28 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  8.8s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.07       TIME AVG. FORCE       0.815    
 LARGEST SCALED RESIDUAL FORCE      5.009E-03   AT NODE      39510   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      5.009E-03
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -9.653E-08   AT NODE        175   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.381E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -4.910E-16   AT NODE      11927   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         3.492E-02   AT NODE      21069   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -9.649E-03   AT NODE      32364   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  8.9s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.07       TIME AVG. FORCE       0.815    
 LARGEST SCALED RESIDUAL FORCE      1.996E-04   AT NODE      39950   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      1.996E-04
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -3.128E-09   AT NODE      16339   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  3.381E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -2.782E-18   AT NODE      42810   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         3.493E-02   AT NODE      21069   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -5.929E-04   AT NODE      14945   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   3.493E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.280    
 STEP TIME COMPLETED        186.    ,  TOTAL TIME COMPLETED        7.386E+03


  INCREMENT    29 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.0s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.08       TIME AVG. FORCE       0.824    
 LARGEST SCALED RESIDUAL FORCE      7.776E-03   AT NODE      39953   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      7.776E-03
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        9.848E-08   AT NODE        253   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.413E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -4.938E-16   AT NODE      11927   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         3.110E-02   AT NODE      15162   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -7.756E-03   AT NODE      21069   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  8.9s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.08       TIME AVG. FORCE       0.824    
 LARGEST SCALED RESIDUAL FORCE      6.680E-05   AT NODE      39972   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      6.680E-05
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -3.326E-09   AT NODE      16339   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  3.413E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -3.114E-18   AT NODE      42810   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         3.110E-02   AT NODE      15162   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -6.057E-04   AT NODE      14945   DOF 11
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

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.290    
 STEP TIME COMPLETED        193.    ,  TOTAL TIME COMPLETED        7.393E+03


  INCREMENT    30 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.08       TIME AVG. FORCE       0.832    
 LARGEST SCALED RESIDUAL FORCE     -2.594E-02   AT NODE      41866   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -2.594E-02
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -1.016E-07   AT NODE        175   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.445E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -4.969E-16   AT NODE      11927   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         2.891E-02   AT NODE      15162   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -6.434E-03   AT NODE      16502   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.3s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.08       TIME AVG. FORCE       0.832    
 LARGEST SCALED RESIDUAL FORCE      3.948E-04   AT NODE      39934   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      3.948E-04
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -3.639E-09   AT NODE      16622   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  3.445E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         1.005E-17   AT NODE      37134   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         2.891E-02   AT NODE      15162   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -6.195E-04   AT NODE      14945   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   2.891E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.300    
 STEP TIME COMPLETED        200.    ,  TOTAL TIME COMPLETED        7.399E+03


  INCREMENT    31 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.2s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.08       TIME AVG. FORCE       0.840    
 LARGEST SCALED RESIDUAL FORCE      1.125E-02   AT NODE      36312   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      1.125E-02
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -1.036E-07   AT NODE        175   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.476E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.011E-16   AT NODE      11927   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         2.479E-02   AT NODE      16503   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -5.736E-03   AT NODE      15163   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.2s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.08       TIME AVG. FORCE       0.840    
 LARGEST SCALED RESIDUAL FORCE      8.814E-05   AT NODE      41158   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      8.814E-05
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -3.842E-09   AT NODE      16622   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  3.476E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -3.753E-18   AT NODE      42766   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         2.479E-02   AT NODE      16503   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -6.344E-04   AT NODE      14945   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   2.479E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.310    
 STEP TIME COMPLETED        206.    ,  TOTAL TIME COMPLETED        7.406E+03


  INCREMENT    32 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.0s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.08       TIME AVG. FORCE       0.848    
 LARGEST SCALED RESIDUAL FORCE     -1.184E-02   AT NODE      37872   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -1.184E-02
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -1.078E-07   AT NODE        175   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.505E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.056E-16   AT NODE      11927   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         2.346E-02   AT NODE      21060   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        3.942E-03   AT NODE      21061   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.3s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.08       TIME AVG. FORCE       0.848    
 LARGEST SCALED RESIDUAL FORCE      9.162E-05   AT NODE      40499   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      9.162E-05
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -4.103E-09   AT NODE      16622   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  3.505E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -4.393E-18   AT NODE      41934   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         2.346E-02   AT NODE      21060   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -6.506E-04   AT NODE      14945   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   2.346E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.320    
 STEP TIME COMPLETED        213.    ,  TOTAL TIME COMPLETED        7.413E+03


  INCREMENT    33 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.09       TIME AVG. FORCE       0.855    
 LARGEST SCALED RESIDUAL FORCE     -8.652E-03   AT NODE      34846   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -8.652E-03
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -1.143E-07   AT NODE          5   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.533E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.110E-16   AT NODE      11927   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         2.201E-02   AT NODE      33246   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -4.520E-03   AT NODE      21069   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  25s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.09       TIME AVG. FORCE       0.855    
 LARGEST SCALED RESIDUAL FORCE      9.630E-05   AT NODE      41474   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      9.630E-05
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -4.356E-09   AT NODE      16622   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  3.533E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -4.297E-18   AT NODE      42766   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         2.201E-02   AT NODE      33246   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -6.689E-04   AT NODE      14945   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   2.201E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.330    
 STEP TIME COMPLETED        219.    ,  TOTAL TIME COMPLETED        7.419E+03


  INCREMENT    34 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  26s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.09       TIME AVG. FORCE       0.862    
 LARGEST SCALED RESIDUAL FORCE      9.945E-03   AT NODE      37626   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      9.945E-03
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -1.183E-07   AT NODE          5   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.559E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.185E-16   AT NODE      11927   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         2.043E-02   AT NODE      33246   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.916E-03   AT NODE      15162   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  26s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.09       TIME AVG. FORCE       0.862    
 LARGEST SCALED RESIDUAL FORCE      1.142E-04   AT NODE      39972   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      1.142E-04
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -4.771E-09   AT NODE      16622   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  3.559E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -4.760E-18   AT NODE      41934   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         2.043E-02   AT NODE      33246   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -6.882E-04   AT NODE      14945   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   2.043E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.340    
 STEP TIME COMPLETED        226.    ,  TOTAL TIME COMPLETED        7.426E+03


  INCREMENT    35 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.2s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.09       TIME AVG. FORCE       0.869    
 LARGEST SCALED RESIDUAL FORCE      3.710E-02   AT NODE      42409   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      3.710E-02
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -1.216E-07   AT NODE          5   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.585E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.263E-16   AT NODE      11927   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         1.932E-02   AT NODE      21070   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.343E-03   AT NODE      16503   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  13s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.09       TIME AVG. FORCE       0.869    
 LARGEST SCALED RESIDUAL FORCE      1.205E-04   AT NODE      39972   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      1.205E-04
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -5.208E-09   AT NODE      16622   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  3.585E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         2.056E-17   AT NODE      42408   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         1.932E-02   AT NODE      21070   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -7.084E-04   AT NODE      14863   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   1.932E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.350    
 STEP TIME COMPLETED        233.    ,  TOTAL TIME COMPLETED        7.433E+03


  INCREMENT    36 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  13s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.09       TIME AVG. FORCE       0.875    
 LARGEST SCALED RESIDUAL FORCE     -5.557E-03   AT NODE      42035   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -5.557E-03
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.254E-07   AT NODE          6   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.610E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.345E-16   AT NODE      11927   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         1.755E-02   AT NODE      21070   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.128E-03   AT NODE      15161   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  11s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.09       TIME AVG. FORCE       0.875    
 LARGEST SCALED RESIDUAL FORCE      1.325E-04   AT NODE      39972   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      1.325E-04
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -5.423E-09   AT NODE      16622   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  3.609E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.382E-18   AT NODE      41968   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         1.756E-02   AT NODE      21070   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -7.295E-04   AT NODE      14863   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   1.756E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.360    
 STEP TIME COMPLETED        239.    ,  TOTAL TIME COMPLETED        7.439E+03


  INCREMENT    37 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.9s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.10       TIME AVG. FORCE       0.881    
 LARGEST SCALED RESIDUAL FORCE     -2.787E-02   AT NODE      42409   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -2.787E-02
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -1.285E-07   AT NODE          5   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.633E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.435E-16   AT NODE      11927   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         1.591E-02   AT NODE      21060   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.531E-03   AT NODE      15161   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.10       TIME AVG. FORCE       0.881    
 LARGEST SCALED RESIDUAL FORCE      1.449E-04   AT NODE      41158   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      1.449E-04
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -5.914E-09   AT NODE      16622   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  3.633E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         2.351E-17   AT NODE      42407   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         1.591E-02   AT NODE      21060   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -7.510E-04   AT NODE      14863   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   1.591E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.370    
 STEP TIME COMPLETED        246.    ,  TOTAL TIME COMPLETED        7.446E+03


  INCREMENT    38 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.10       TIME AVG. FORCE       0.887    
 LARGEST SCALED RESIDUAL FORCE      5.545E-02   AT NODE      43133   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      5.545E-02
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -1.345E-07   AT NODE          5   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.657E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.519E-16   AT NODE      11927   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         1.494E-02   AT NODE      21061   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.586E-03   AT NODE      30580   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.3s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.10       TIME AVG. FORCE       0.887    
 LARGEST SCALED RESIDUAL FORCE     -5.344E-04   AT NODE      43122   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -5.344E-04
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        6.165E-09   AT NODE      21113   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  3.657E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         3.476E-17   AT NODE      43132   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         1.494E-02   AT NODE      21061   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -7.737E-04   AT NODE      14863   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   1.494E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.380    
 STEP TIME COMPLETED        253.    ,  TOTAL TIME COMPLETED        7.453E+03


  INCREMENT    39 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  12s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.10       TIME AVG. FORCE       0.892    
 LARGEST SCALED RESIDUAL FORCE     -8.191E-03   AT NODE      43144   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -8.191E-03
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -1.376E-07   AT NODE          5   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.679E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.622E-16   AT NODE      11927   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         1.409E-02   AT NODE      21061   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.340E-03   AT NODE      33246   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  27s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.10       TIME AVG. FORCE       0.892    
 LARGEST SCALED RESIDUAL FORCE      1.732E-04   AT NODE      41158   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      1.732E-04
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        6.644E-09   AT NODE      14695   DOF  1
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  3.679E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.067E-17   AT NODE      43155   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.         1.409E-02   AT NODE      21061   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -7.970E-04   AT NODE      14863   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS   1.409E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.390    
 STEP TIME COMPLETED        259.    ,  TOTAL TIME COMPLETED        7.459E+03


  INCREMENT    40 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  28s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.10       TIME AVG. FORCE       0.897    
 LARGEST SCALED RESIDUAL FORCE     -3.951E-03   AT NODE      39825   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -3.951E-03
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -1.441E-07   AT NODE          5   DOF  1
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  3.701E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.716E-16   AT NODE      11927   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -1.439E-02   AT NODE      16301   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.677E-03   AT NODE      20173   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -1.439E-02

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.400    
 STEP TIME COMPLETED        266.    ,  TOTAL TIME COMPLETED        7.466E+03


  INCREMENT    41 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  27s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.10       TIME AVG. FORCE       0.902    
 LARGEST SCALED RESIDUAL FORCE      2.020E-02   AT NODE      43122   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      2.020E-02
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -1.393E-07   AT NODE          5   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.723E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.898E-16   AT NODE      11927   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -1.500E-02   AT NODE      16301   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.265E-03   AT NODE      20055   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  27s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.10       TIME AVG. FORCE       0.902    
 LARGEST SCALED RESIDUAL FORCE      1.954E-04   AT NODE      39972   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      1.954E-04
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        7.540E-09   AT NODE      14695   DOF  1
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  3.723E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.525E-17   AT NODE      43111   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -1.504E-02   AT NODE      16301   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -8.505E-04   AT NODE      14863   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -1.504E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.410    
 STEP TIME COMPLETED        273.    ,  TOTAL TIME COMPLETED        7.473E+03


  INCREMENT    42 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  27s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.10       TIME AVG. FORCE       0.907    
 LARGEST SCALED RESIDUAL FORCE      1.289E-02   AT NODE      42385   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      1.289E-02
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -1.561E-07   AT NODE          5   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.745E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.931E-16   AT NODE      11927   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -1.504E-02   AT NODE      16295   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.606E-03   AT NODE      21061   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  27s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.10       TIME AVG. FORCE       0.907    
 LARGEST SCALED RESIDUAL FORCE      2.135E-04   AT NODE      41158   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      2.135E-04
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        8.160E-09   AT NODE      14695   DOF  1
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  3.745E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -9.756E-18   AT NODE      42373   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -1.510E-02   AT NODE      16295   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -8.734E-04   AT NODE      14863   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -1.510E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.420    
 STEP TIME COMPLETED        279.    ,  TOTAL TIME COMPLETED        7.479E+03


  INCREMENT    43 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  27s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.10       TIME AVG. FORCE       0.912    
 LARGEST SCALED RESIDUAL FORCE     -4.691E-03   AT NODE      39813   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -4.691E-03
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -1.574E-07   AT NODE          5   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.767E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -6.088E-16   AT NODE      11926   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -1.562E-02   AT NODE      16295   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.728E-03   AT NODE       6333   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  17s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.10       TIME AVG. FORCE       0.912    
 LARGEST SCALED RESIDUAL FORCE      2.301E-04   AT NODE      39972   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      2.301E-04
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        8.774E-09   AT NODE      14695   DOF  1
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  3.767E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -8.766E-18   AT NODE      43111   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -1.568E-02   AT NODE      16295   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -9.028E-04   AT NODE      14863   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -1.568E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.430    
 STEP TIME COMPLETED        286.    ,  TOTAL TIME COMPLETED        7.486E+03


  INCREMENT    44 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.2s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.10       TIME AVG. FORCE       0.916    
 LARGEST SCALED RESIDUAL FORCE      7.088E-03   AT NODE      37980   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      7.088E-03
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -1.621E-07   AT NODE          5   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.789E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -6.272E-16   AT NODE      11926   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -1.615E-02   AT NODE      16295   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.050E-03   AT NODE      21070   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.2s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.10       TIME AVG. FORCE       0.916    
 LARGEST SCALED RESIDUAL FORCE      2.513E-04   AT NODE      40499   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      2.513E-04
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        9.465E-09   AT NODE      14695   DOF  1
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  3.789E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -9.718E-18   AT NODE      43166   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -1.620E-02   AT NODE      16295   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -9.315E-04   AT NODE      14863   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -1.620E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.440    
 STEP TIME COMPLETED        293.    ,  TOTAL TIME COMPLETED        7.493E+03


  INCREMENT    45 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  8.9s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.920    
 LARGEST SCALED RESIDUAL FORCE      1.173E-02   AT NODE      39832   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      1.173E-02
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -1.659E-07   AT NODE          5   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.812E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -6.461E-16   AT NODE      11926   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -1.667E-02   AT NODE      16295   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.400E-03   AT NODE      37244   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.920    
 LARGEST SCALED RESIDUAL FORCE      5.518E-04   AT NODE      39946   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      5.518E-04
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.018E-08   AT NODE      14695   DOF  1
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  3.811E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -9.724E-18   AT NODE      43111   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -1.672E-02   AT NODE      16295   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -9.648E-04   AT NODE      14862   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -1.672E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.450    
 STEP TIME COMPLETED        299.    ,  TOTAL TIME COMPLETED        7.499E+03


  INCREMENT    46 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.924    
 LARGEST SCALED RESIDUAL FORCE      1.423E-02   AT NODE      39849   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      1.423E-02
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -1.686E-07   AT NODE          5   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.834E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -6.650E-16   AT NODE      11926   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -1.720E-02   AT NODE      16783   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.263E-03   AT NODE      37604   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.0s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.924    
 LARGEST SCALED RESIDUAL FORCE      2.904E-04   AT NODE      39972   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      2.904E-04
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.099E-08   AT NODE      14695   DOF  1
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  3.834E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.314E-17   AT NODE      43099   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -1.725E-02   AT NODE      16783   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -9.999E-04   AT NODE      14862   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -1.725E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.460    
 STEP TIME COMPLETED        306.    ,  TOTAL TIME COMPLETED        7.506E+03


  INCREMENT    47 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.0s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.928    
 LARGEST SCALED RESIDUAL FORCE     -5.484E-03   AT NODE      39825   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -5.484E-03
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -1.752E-07   AT NODE          5   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.857E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -6.854E-16   AT NODE      11926   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -1.773E-02   AT NODE      16783   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.338E-03   AT NODE      37750   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.4s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.928    
 LARGEST SCALED RESIDUAL FORCE      3.035E-04   AT NODE      39972   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      3.035E-04
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.154E-08   AT NODE      14695   DOF  1
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  3.857E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.185E-17   AT NODE      43177   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -1.777E-02   AT NODE      16783   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.037E-03   AT NODE      14862   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -1.777E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.470    
 STEP TIME COMPLETED        313.    ,  TOTAL TIME COMPLETED        7.513E+03


  INCREMENT    48 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.932    
 LARGEST SCALED RESIDUAL FORCE     -5.719E-03   AT NODE      39825   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -5.719E-03
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -1.814E-07   AT NODE          5   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.878E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -7.073E-16   AT NODE      11926   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -1.835E-02   AT NODE      16783   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.821E-03   AT NODE      35616   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.4s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.932    
 LARGEST SCALED RESIDUAL FORCE      3.215E-04   AT NODE      39972   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      3.215E-04
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.228E-08   AT NODE      14695   DOF  1
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  3.878E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.272E-17   AT NODE      43177   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -1.840E-02   AT NODE      16783   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.076E-03   AT NODE      14862   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -1.840E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.480    
 STEP TIME COMPLETED        319.    ,  TOTAL TIME COMPLETED        7.519E+03


  INCREMENT    49 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.2s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.936    
 LARGEST SCALED RESIDUAL FORCE     -1.055E-02   AT NODE      39849   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -1.055E-02
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -1.862E-07   AT NODE          5   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.900E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -7.301E-16   AT NODE      11926   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -1.906E-02   AT NODE      16783   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.390E-03   AT NODE      14862   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.2s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.936    
 LARGEST SCALED RESIDUAL FORCE     -9.346E-04   AT NODE      42383   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -9.346E-04
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.311E-08   AT NODE      14695   DOF  1
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  3.900E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.363E-17   AT NODE      43176   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -1.911E-02   AT NODE      16783   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.118E-03   AT NODE      14862   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -1.911E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.490    
 STEP TIME COMPLETED        326.    ,  TOTAL TIME COMPLETED        7.526E+03


  INCREMENT    50 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  8.9s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.939    
 LARGEST SCALED RESIDUAL FORCE     -1.000E-02   AT NODE      42376   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -1.000E-02
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -1.905E-07   AT NODE          5   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.922E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -7.536E-16   AT NODE      11926   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -1.986E-02   AT NODE      20913   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.439E-03   AT NODE      14862   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.5s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.939    
 LARGEST SCALED RESIDUAL FORCE      8.148E-04   AT NODE      42460   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      8.148E-04
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.416E-08   AT NODE      14695   DOF  1
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  3.921E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.477E-17   AT NODE      43100   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -1.992E-02   AT NODE      20913   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.160E-03   AT NODE      14862   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -1.992E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.500    
 STEP TIME COMPLETED        332.    ,  TOTAL TIME COMPLETED        7.532E+03


  INCREMENT    51 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.4s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.942    
 LARGEST SCALED RESIDUAL FORCE     -6.325E-03   AT NODE      39825   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -6.325E-03
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -1.958E-07   AT NODE          5   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.944E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -7.782E-16   AT NODE      11926   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -2.076E-02   AT NODE      20913   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.482E-03   AT NODE      14862   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.4s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.942    
 LARGEST SCALED RESIDUAL FORCE     -4.840E-04   AT NODE      40914   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -4.840E-04
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.484E-08   AT NODE      14695   DOF  1
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  3.943E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.603E-17   AT NODE      43100   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -2.082E-02   AT NODE      20913   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.204E-03   AT NODE      14862   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -2.082E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.510    
 STEP TIME COMPLETED        339.    ,  TOTAL TIME COMPLETED        7.539E+03


  INCREMENT    52 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.3s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.946    
 LARGEST SCALED RESIDUAL FORCE     -1.298E-02   AT NODE      40003   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -1.298E-02
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -2.004E-07   AT NODE          5   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.967E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -8.044E-16   AT NODE      11925   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -2.176E-02   AT NODE      20913   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.541E-03   AT NODE      14862   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.3s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.946    
 LARGEST SCALED RESIDUAL FORCE     -9.537E-04   AT NODE      43168   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -9.537E-04
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.578E-08   AT NODE      14695   DOF  1
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  3.966E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.685E-17   AT NODE      43177   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -2.182E-02   AT NODE      20913   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.250E-03   AT NODE      14862   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -2.182E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.520    
 STEP TIME COMPLETED        346.    ,  TOTAL TIME COMPLETED        7.546E+03


  INCREMENT    53 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.2s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.949    
 LARGEST SCALED RESIDUAL FORCE     -9.343E-03   AT NODE      40816   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -9.343E-03
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -2.059E-07   AT NODE          5   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  3.991E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -8.366E-16   AT NODE      11925   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -2.277E-02   AT NODE      20913   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.583E-03   AT NODE      14862   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  8.9s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.949    
 LARGEST SCALED RESIDUAL FORCE      1.555E-03   AT NODE      43164   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      1.555E-03
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.664E-08   AT NODE      14695   DOF  1
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  3.990E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.840E-17   AT NODE      43177   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -2.282E-02   AT NODE      20913   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.297E-03   AT NODE      14862   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -2.282E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.530    
 STEP TIME COMPLETED        352.    ,  TOTAL TIME COMPLETED        7.552E+03


  INCREMENT    54 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.3s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.952    
 LARGEST SCALED RESIDUAL FORCE     -6.704E-03   AT NODE      39825   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -6.704E-03
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -2.109E-07   AT NODE          5   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  4.016E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -8.702E-16   AT NODE      11925   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -2.395E-02   AT NODE      16296   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.639E-03   AT NODE      14862   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.952    
 LARGEST SCALED RESIDUAL FORCE      4.234E-04   AT NODE      40499   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      4.234E-04
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.746E-08   AT NODE      14695   DOF  1
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  4.015E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.870E-17   AT NODE      43100   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -2.404E-02   AT NODE      16296   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.347E-03   AT NODE      14862   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -2.404E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.540    
 STEP TIME COMPLETED        359.    ,  TOTAL TIME COMPLETED        7.559E+03


  INCREMENT    55 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.0s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.955    
 LARGEST SCALED RESIDUAL FORCE     -6.779E-03   AT NODE      39825   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -6.779E-03
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -2.146E-07   AT NODE          5   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  4.042E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -9.052E-16   AT NODE      11925   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -2.537E-02   AT NODE      16296   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.683E-03   AT NODE      14862   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.955    
 LARGEST SCALED RESIDUAL FORCE     -3.080E-03   AT NODE      41394   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -3.080E-03
 LARGEST INCREMENT OF DISP.         1.662E-05   AT NODE      42473   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.829E-08   AT NODE      14695   DOF  1
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  4.041E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.985E-17   AT NODE      43100   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -2.545E-02   AT NODE      16296   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.398E-03   AT NODE      14862   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -2.545E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.550    
 STEP TIME COMPLETED        366.    ,  TOTAL TIME COMPLETED        7.566E+03


  INCREMENT    56 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.3s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.957    
 LARGEST SCALED RESIDUAL FORCE     -0.219       AT NODE      42998   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -0.219    
 LARGEST INCREMENT OF DISP.         1.664E-05   AT NODE      40409   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -2.175E-07   AT NODE          5   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  4.068E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -9.410E-16   AT NODE      11925   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -2.684E-02   AT NODE      16296   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.728E-03   AT NODE      14861   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.5s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.957    
 LARGEST SCALED RESIDUAL FORCE     -2.222E-03   AT NODE      41730   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -2.222E-03
 LARGEST INCREMENT OF DISP.         1.663E-05   AT NODE      40409   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.105E-08   AT NODE      14695   DOF  1
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  4.067E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -6.143E-17   AT NODE      43177   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -2.692E-02   AT NODE      16296   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.450E-03   AT NODE      14862   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -2.692E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.560    
 STEP TIME COMPLETED        372.    ,  TOTAL TIME COMPLETED        7.572E+03


  INCREMENT    57 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.2s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.960    
 LARGEST SCALED RESIDUAL FORCE     -4.613E-02   AT NODE      41112   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -4.613E-02
 LARGEST INCREMENT OF DISP.         1.665E-05   AT NODE      40409   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -2.161E-07   AT NODE          5   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  4.096E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -9.759E-16   AT NODE      11925   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -2.834E-02   AT NODE      16296   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.774E-03   AT NODE      14861   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.2s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.960    
 LARGEST SCALED RESIDUAL FORCE      6.504E-03   AT NODE      37102   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      6.504E-03
 LARGEST INCREMENT OF DISP.         1.664E-05   AT NODE      40409   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.991E-08   AT NODE      14695   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  4.095E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -2.541E-17   AT NODE      42032   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -2.842E-02   AT NODE      16296   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.500E-03   AT NODE      14862   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.960    
 LARGEST SCALED RESIDUAL FORCE     -7.156E-04   AT NODE      39875   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -7.156E-04
 LARGEST INCREMENT OF DISP.         1.664E-05   AT NODE      40409   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -2.870E-10   AT NODE      15706   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  4.095E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         8.788E-18   AT NODE      39908   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -2.842E-02   AT NODE      16296   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        5.587E-06   AT NODE      16299   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -2.842E-02

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.570    
 STEP TIME COMPLETED        379.    ,  TOTAL TIME COMPLETED        7.579E+03


  INCREMENT    58 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.3s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.963    
 LARGEST SCALED RESIDUAL FORCE     -0.124       AT NODE      39906   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -0.124    
 LARGEST INCREMENT OF DISP.         1.666E-05   AT NODE         26   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -2.146E-07   AT NODE          5   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  4.124E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.011E-15   AT NODE      11925   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -2.983E-02   AT NODE      16296   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.769E-03   AT NODE      14861   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.963    
 LARGEST SCALED RESIDUAL FORCE     -2.391E-02   AT NODE      39906   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -2.391E-02
 LARGEST INCREMENT OF DISP.         1.666E-05   AT NODE      40409   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.098E-08   AT NODE      14695   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  4.124E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         8.309E-17   AT NODE      40884   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -2.990E-02   AT NODE      16296   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.551E-03   AT NODE      14862   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.2s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.963    
 LARGEST SCALED RESIDUAL FORCE      1.269E-03   AT NODE      39849   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      1.269E-03
 LARGEST INCREMENT OF DISP.         1.666E-05   AT NODE      40409   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -9.196E-10   AT NODE      15500   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  4.124E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         2.069E-17   AT NODE      40917   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -2.991E-02   AT NODE      16296   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        6.211E-06   AT NODE      16293   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -2.991E-02

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.580    
 STEP TIME COMPLETED        386.    ,  TOTAL TIME COMPLETED        7.586E+03


  INCREMENT    59 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.965    
 LARGEST SCALED RESIDUAL FORCE     -6.218E-02   AT NODE      40499   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -6.218E-02
 LARGEST INCREMENT OF DISP.         1.668E-05   AT NODE         26   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -2.097E-07   AT NODE          5   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  4.154E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.044E-15   AT NODE      11925   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.132E-02   AT NODE      16296   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.854E-03   AT NODE      14861   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.4s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.965    
 LARGEST SCALED RESIDUAL FORCE     -1.356E-02   AT NODE      41019   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -1.356E-02
 LARGEST INCREMENT OF DISP.         1.667E-05   AT NODE         26   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.101E-08   AT NODE      14695   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  4.153E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -2.102E-17   AT NODE      42733   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.138E-02   AT NODE      16296   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.598E-03   AT NODE      14862   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.965    
 LARGEST SCALED RESIDUAL FORCE      3.753E-04   AT NODE      40918   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      3.753E-04
 LARGEST INCREMENT OF DISP.         1.667E-05   AT NODE         26   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -3.070E-10   AT NODE      17464   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  4.153E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.444E-18   AT NODE      40815   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.138E-02   AT NODE      16296   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        6.304E-06   AT NODE      16293   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -3.138E-02

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.590    
 STEP TIME COMPLETED        392.    ,  TOTAL TIME COMPLETED        7.592E+03


  INCREMENT    60 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.967    
 LARGEST SCALED RESIDUAL FORCE     -4.191E-02   AT NODE      40707   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -4.191E-02
 LARGEST INCREMENT OF DISP.         1.669E-05   AT NODE         26   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.062E-07   AT NODE      16274   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  4.185E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.076E-15   AT NODE      11925   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.274E-02   AT NODE      16296   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.856E-03   AT NODE      14861   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.3s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.967    
 LARGEST SCALED RESIDUAL FORCE     -1.580E-03   AT NODE      39819   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -1.580E-03
 LARGEST INCREMENT OF DISP.         1.669E-05   AT NODE         26   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.140E-08   AT NODE      14695   DOF  1
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  4.184E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -3.172E-17   AT NODE      40815   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.280E-02   AT NODE      16296   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.649E-03   AT NODE      14861   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -3.280E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.600    
 STEP TIME COMPLETED        399.    ,  TOTAL TIME COMPLETED        7.599E+03


  INCREMENT    61 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.4s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.970    
 LARGEST SCALED RESIDUAL FORCE     -6.513E-02   AT NODE      40308   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -6.513E-02
 LARGEST INCREMENT OF DISP.         1.671E-05   AT NODE         26   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.042E-07   AT NODE      16274   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  4.218E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.110E-15   AT NODE      11924   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.412E-02   AT NODE      16296   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.916E-03   AT NODE      14861   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  12s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.970    
 LARGEST SCALED RESIDUAL FORCE     -2.104E-02   AT NODE      41357   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -2.104E-02
 LARGEST INCREMENT OF DISP.         1.670E-05   AT NODE         26   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.173E-08   AT NODE      14695   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  4.217E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -3.186E-17   AT NODE      40815   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.417E-02   AT NODE      16296   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.701E-03   AT NODE      14861   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  26s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.970    
 LARGEST SCALED RESIDUAL FORCE     -1.515E-03   AT NODE      41356   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -1.515E-03
 LARGEST INCREMENT OF DISP.         1.670E-05   AT NODE         26   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        3.166E-10   AT NODE      37142   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  4.217E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -4.837E-18   AT NODE      41312   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.417E-02   AT NODE      16296   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        7.043E-06   AT NODE      16293   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -3.417E-02

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.610    
 STEP TIME COMPLETED        406.    ,  TOTAL TIME COMPLETED        7.606E+03


  INCREMENT    62 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  24s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.972    
 LARGEST SCALED RESIDUAL FORCE     -8.445E-02   AT NODE      41357   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -8.445E-02
 LARGEST INCREMENT OF DISP.         1.672E-05   AT NODE       1817   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.007E-07   AT NODE      16268   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  4.252E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.147E-15   AT NODE      11924   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.537E-02   AT NODE      16296   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.827E-03   AT NODE      14860   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.972    
 LARGEST SCALED RESIDUAL FORCE     -4.777E-03   AT NODE      39895   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -4.777E-03
 LARGEST INCREMENT OF DISP.         1.672E-05   AT NODE       1817   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.128E-08   AT NODE      14695   DOF  1
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  4.251E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -2.632E-17   AT NODE      41831   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.542E-02   AT NODE      16296   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.752E-03   AT NODE      14861   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -3.542E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.620    
 STEP TIME COMPLETED        412.    ,  TOTAL TIME COMPLETED        7.612E+03


  INCREMENT    63 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.0s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.974    
 LARGEST SCALED RESIDUAL FORCE     -1.980E-02   AT NODE      37054   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -1.980E-02
 LARGEST INCREMENT OF DISP.         1.674E-05   AT NODE         26   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.008E-07   AT NODE      16281   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  4.287E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.183E-15   AT NODE      11924   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.654E-02   AT NODE      16296   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.962E-03   AT NODE      14860   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.0s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.974    
 LARGEST SCALED RESIDUAL FORCE     -9.846E-04   AT NODE      13384   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -9.846E-04
 LARGEST INCREMENT OF DISP.         1.673E-05   AT NODE         26   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.042E-08   AT NODE      14695   DOF  1
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  4.286E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -2.560E-17   AT NODE      42733   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.659E-02   AT NODE      16296   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.799E-03   AT NODE      14861   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -3.659E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.630    
 STEP TIME COMPLETED        419.    ,  TOTAL TIME COMPLETED        7.619E+03


  INCREMENT    64 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.3s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.976    
 LARGEST SCALED RESIDUAL FORCE     -1.477E-02   AT NODE      40589   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -1.477E-02
 LARGEST INCREMENT OF DISP.         1.675E-05   AT NODE       1817   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.006E-07   AT NODE      16281   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  4.323E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.219E-15   AT NODE      11924   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.760E-02   AT NODE      16296   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.910E-03   AT NODE      14860   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.0s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.976    
 LARGEST SCALED RESIDUAL FORCE     -1.788E-03   AT NODE       5848   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -1.788E-03
 LARGEST INCREMENT OF DISP.         1.675E-05   AT NODE       1817   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.106E-08   AT NODE      14695   DOF  1
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  4.323E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -2.678E-17   AT NODE      42733   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.765E-02   AT NODE      16296   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.844E-03   AT NODE      14861   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -3.765E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.640    
 STEP TIME COMPLETED        426.    ,  TOTAL TIME COMPLETED        7.626E+03


  INCREMENT    65 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.0s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.978    
 LARGEST SCALED RESIDUAL FORCE     -2.294E-02   AT NODE      40699   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -2.294E-02
 LARGEST INCREMENT OF DISP.         1.677E-05   AT NODE      16301   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.974E-07   AT NODE      16281   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  4.360E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.251E-15   AT NODE      11924   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.869E-02   AT NODE      16296   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.939E-03   AT NODE      14860   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.978    
 LARGEST SCALED RESIDUAL FORCE     -7.244E-03   AT NODE      39840   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -7.244E-03
 LARGEST INCREMENT OF DISP.         1.676E-05   AT NODE      14935   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.167E-08   AT NODE      14695   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  4.360E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -2.813E-17   AT NODE      42733   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.873E-02   AT NODE      16296   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.886E-03   AT NODE      14861   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.4s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.978    
 LARGEST SCALED RESIDUAL FORCE      8.578E-04   AT NODE      39907   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      8.578E-04
 LARGEST INCREMENT OF DISP.         1.676E-05   AT NODE      14935   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.177E-09   AT NODE          5   DOF  1
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  4.360E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         7.246E-18   AT NODE      40952   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.874E-02   AT NODE      16296   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        8.163E-06   AT NODE      21088   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -3.874E-02

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.650    
 STEP TIME COMPLETED        432.    ,  TOTAL TIME COMPLETED        7.632E+03


  INCREMENT    66 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.3s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.980    
 LARGEST SCALED RESIDUAL FORCE     -6.319E-02   AT NODE      37077   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -6.319E-02
 LARGEST INCREMENT OF DISP.         1.679E-05   AT NODE      16301   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.939E-07   AT NODE      16281   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  4.399E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.281E-15   AT NODE      11924   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.993E-02   AT NODE      21129   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.882E-03   AT NODE      14860   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.2s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.980    
 LARGEST SCALED RESIDUAL FORCE     -6.222E-03   AT NODE      38006   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -6.222E-03
 LARGEST INCREMENT OF DISP.         1.678E-05   AT NODE      16301   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.284E-08   AT NODE      14695   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  4.398E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -3.749E-17   AT NODE      37137   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.001E-02   AT NODE      21129   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.924E-03   AT NODE      14861   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  8.9s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.980    
 LARGEST SCALED RESIDUAL FORCE     -8.844E-04   AT NODE      35207   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -8.844E-04
 LARGEST INCREMENT OF DISP.         1.678E-05   AT NODE      16301   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -9.937E-10   AT NODE          6   DOF  1
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  4.398E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX         7.675E-18   AT NODE      37134   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.001E-02   AT NODE      21129   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        8.832E-06   AT NODE      21088   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -4.001E-02

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.660    
 STEP TIME COMPLETED        439.    ,  TOTAL TIME COMPLETED        7.639E+03


  INCREMENT    67 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.982    
 LARGEST SCALED RESIDUAL FORCE     -1.735E-02   AT NODE      39925   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -1.735E-02
 LARGEST INCREMENT OF DISP.         1.681E-05   AT NODE      16295   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.917E-07   AT NODE      16275   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  4.438E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.308E-15   AT NODE      11924   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.143E-02   AT NODE      21129   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.942E-03   AT NODE      14859   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.2s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.982    
 LARGEST SCALED RESIDUAL FORCE     -3.759E-03   AT NODE      37565   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -3.759E-03
 LARGEST INCREMENT OF DISP.         1.680E-05   AT NODE      16302   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.994E-08   AT NODE      14695   DOF  1
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  4.437E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -3.030E-17   AT NODE      42733   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.151E-02   AT NODE      21129   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.956E-03   AT NODE      14861   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -4.151E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.670    
 STEP TIME COMPLETED        446.    ,  TOTAL TIME COMPLETED        7.646E+03


  INCREMENT    68 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.984    
 LARGEST SCALED RESIDUAL FORCE      2.203E-02   AT NODE      39961   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      2.203E-02
 LARGEST INCREMENT OF DISP.         1.682E-05   AT NODE      16302   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.904E-07   AT NODE      16275   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  4.480E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.333E-15   AT NODE      11923   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.294E-02   AT NODE      21129   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.972E-03   AT NODE      14859   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.11       TIME AVG. FORCE       0.984    
 LARGEST SCALED RESIDUAL FORCE      2.103E-03   AT NODE      39870   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      2.103E-03
 LARGEST INCREMENT OF DISP.         1.682E-05   AT NODE      16302   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.871E-08   AT NODE      14695   DOF  1
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  4.479E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -3.127E-17   AT NODE      42733   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.300E-02   AT NODE      21129   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -1.984E-03   AT NODE      14861   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -4.300E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.680    
 STEP TIME COMPLETED        452.    ,  TOTAL TIME COMPLETED        7.652E+03


  INCREMENT    69 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.8s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.10       TIME AVG. FORCE       0.986    
 LARGEST SCALED RESIDUAL FORCE      1.668E-02   AT NODE      39927   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      1.668E-02
 LARGEST INCREMENT OF DISP.         1.684E-05   AT NODE      16296   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.905E-07   AT NODE      16275   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  4.524E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.366E-15   AT NODE      11923   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.430E-02   AT NODE      21129   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.303E-03   AT NODE      21362   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.10       TIME AVG. FORCE       0.986    
 LARGEST SCALED RESIDUAL FORCE     -2.085E-03   AT NODE      37164   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -2.085E-03
 LARGEST INCREMENT OF DISP.         1.683E-05   AT NODE      16296   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.860E-08   AT NODE      14695   DOF  1
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  4.523E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -3.244E-17   AT NODE      42733   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.436E-02   AT NODE      21129   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.011E-03   AT NODE      14861   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -4.436E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.690    
 STEP TIME COMPLETED        459.    ,  TOTAL TIME COMPLETED        7.659E+03


  INCREMENT    70 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.0s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.10       TIME AVG. FORCE       0.987    
 LARGEST SCALED RESIDUAL FORCE     -1.655E-02   AT NODE      40699   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -1.655E-02
 LARGEST INCREMENT OF DISP.         1.686E-05   AT NODE      16296   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.898E-07   AT NODE      16269   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  4.571E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.397E-15   AT NODE      11923   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.555E-02   AT NODE      21129   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.447E-03   AT NODE      21362   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.5s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.10       TIME AVG. FORCE       0.987    
 LARGEST SCALED RESIDUAL FORCE      1.644E-03   AT NODE      34865   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      1.644E-03
 LARGEST INCREMENT OF DISP.         1.685E-05   AT NODE      16296   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.838E-08   AT NODE      14695   DOF  1
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  4.570E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -3.354E-17   AT NODE      42733   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.561E-02   AT NODE      21129   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.040E-03   AT NODE      14860   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -4.561E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.700    
 STEP TIME COMPLETED        465.    ,  TOTAL TIME COMPLETED        7.665E+03


  INCREMENT    71 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  13s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.10       TIME AVG. FORCE       0.989    
 LARGEST SCALED RESIDUAL FORCE      8.596E-03   AT NODE      34865   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      8.596E-03
 LARGEST INCREMENT OF DISP.         1.687E-05   AT NODE      16296   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.907E-07   AT NODE      16269   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  4.623E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.425E-15   AT NODE      11923   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.682E-02   AT NODE      21129   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.815E-03   AT NODE      16301   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.3s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.10       TIME AVG. FORCE       0.989    
 LARGEST SCALED RESIDUAL FORCE      1.572E-03   AT NODE      36587   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      1.572E-03
 LARGEST INCREMENT OF DISP.         1.687E-05   AT NODE      16296   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.827E-08   AT NODE      14695   DOF  1
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  4.622E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -3.463E-17   AT NODE      42733   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.687E-02   AT NODE      21129   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.071E-03   AT NODE      14860   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -4.687E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.710    
 STEP TIME COMPLETED        472.    ,  TOTAL TIME COMPLETED        7.672E+03


  INCREMENT    72 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.2s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.10       TIME AVG. FORCE       0.991    
 LARGEST SCALED RESIDUAL FORCE      1.453E-02   AT NODE      39803   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      1.453E-02
 LARGEST INCREMENT OF DISP.         1.689E-05   AT NODE      16290   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.911E-07   AT NODE      16269   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  4.678E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.451E-15   AT NODE      11923   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.753E-02   AT NODE      21129   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.750E-03   AT NODE      16301   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.4s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.10       TIME AVG. FORCE       0.991    
 LARGEST SCALED RESIDUAL FORCE      1.113E-03   AT NODE      12862   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      1.113E-03
 LARGEST INCREMENT OF DISP.         1.688E-05   AT NODE      16731   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.814E-08   AT NODE      14695   DOF  1
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  4.677E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -3.568E-17   AT NODE      42733   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.758E-02   AT NODE      21129   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.099E-03   AT NODE      14860   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -4.758E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.720    
 STEP TIME COMPLETED        479.    ,  TOTAL TIME COMPLETED        7.679E+03


  INCREMENT    73 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.3s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.10       TIME AVG. FORCE       0.992    
 LARGEST SCALED RESIDUAL FORCE      3.576E-02   AT NODE      37157   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      3.576E-02
 LARGEST INCREMENT OF DISP.         1.690E-05   AT NODE      16297   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.906E-07   AT NODE      16269   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  4.729E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.474E-15   AT NODE      11923   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.801E-02   AT NODE      16291   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.804E-03   AT NODE      16783   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  8.8s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.10       TIME AVG. FORCE       0.992    
 LARGEST SCALED RESIDUAL FORCE      3.351E-03   AT NODE      39871   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      3.351E-03
 LARGEST INCREMENT OF DISP.         1.689E-05   AT NODE      16731   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.836E-08   AT NODE      14695   DOF  1
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  4.728E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -3.793E-17   AT NODE      40850   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.807E-02   AT NODE      16291   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.122E-03   AT NODE      14860   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -4.807E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.730    
 STEP TIME COMPLETED        485.    ,  TOTAL TIME COMPLETED        7.685E+03


  INCREMENT    74 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.3s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.10       TIME AVG. FORCE       0.993    
 LARGEST SCALED RESIDUAL FORCE     -4.613E-02   AT NODE      41474   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -4.613E-02
 LARGEST INCREMENT OF DISP.         1.692E-05   AT NODE      16297   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.912E-07   AT NODE      16263   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  4.781E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.496E-15   AT NODE      11922   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.882E-02   AT NODE      16291   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.955E-03   AT NODE      16295   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  8.9s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.10       TIME AVG. FORCE       0.993    
 LARGEST SCALED RESIDUAL FORCE     -1.902E-02   AT NODE      37137   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -1.902E-02
 LARGEST INCREMENT OF DISP.         1.691E-05   AT NODE      16297   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.773E-08   AT NODE      14695   DOF  1
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  4.780E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -4.341E-17   AT NODE      42002   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.888E-02   AT NODE      16291   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.141E-03   AT NODE      14860   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.2s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.10       TIME AVG. FORCE       0.993    
 LARGEST SCALED RESIDUAL FORCE      2.419E-04   AT NODE      39920   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      2.419E-04
 LARGEST INCREMENT OF DISP.         1.691E-05   AT NODE      16297   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -2.885E-10   AT NODE          6   DOF  1
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  4.780E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -2.622E-18   AT NODE      37134   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.888E-02   AT NODE      16291   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.209E-05   AT NODE      16288   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -4.888E-02

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.740    
 STEP TIME COMPLETED        492.    ,  TOTAL TIME COMPLETED        7.692E+03


  INCREMENT    75 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.09       TIME AVG. FORCE       0.995    
 LARGEST SCALED RESIDUAL FORCE      4.743E-02   AT NODE      37158   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      4.743E-02
 LARGEST INCREMENT OF DISP.         1.693E-05   AT NODE      16297   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.921E-07   AT NODE      16263   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  4.833E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.525E-15   AT NODE      11922   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.916E-02   AT NODE      16678   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.921E-03   AT NODE      20913   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.09       TIME AVG. FORCE       0.995    
 LARGEST SCALED RESIDUAL FORCE     -2.712E-03   AT NODE      37206   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -2.712E-03
 LARGEST INCREMENT OF DISP.         1.692E-05   AT NODE      16297   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.808E-08   AT NODE      14695   DOF  1
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  4.832E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -3.906E-17   AT NODE      42733   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.921E-02   AT NODE      16678   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.157E-03   AT NODE      14860   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -4.921E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.750    
 STEP TIME COMPLETED        499.    ,  TOTAL TIME COMPLETED        7.699E+03


  INCREMENT    76 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  8.9s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.09       TIME AVG. FORCE       0.996    
 LARGEST SCALED RESIDUAL FORCE      2.520E-02   AT NODE      37207   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      2.520E-02
 LARGEST INCREMENT OF DISP.         1.694E-05   AT NODE      16678   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.912E-07   AT NODE      16263   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  4.887E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.550E-15   AT NODE      11922   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.905E-02   AT NODE      16678   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.793E-03   AT NODE      20912   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.2s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.09       TIME AVG. FORCE       0.996    
 LARGEST SCALED RESIDUAL FORCE     -2.521E-03   AT NODE      37089   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -2.521E-03
 LARGEST INCREMENT OF DISP.         1.694E-05   AT NODE      16678   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.754E-08   AT NODE      14695   DOF  1
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  4.886E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -4.026E-17   AT NODE      42733   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.910E-02   AT NODE      16678   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.165E-03   AT NODE      14860   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -4.910E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.760    
 STEP TIME COMPLETED        505.    ,  TOTAL TIME COMPLETED        7.705E+03


  INCREMENT    77 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.0s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.09       TIME AVG. FORCE       0.997    
 LARGEST SCALED RESIDUAL FORCE      1.906E-02   AT NODE      39919   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      1.906E-02
 LARGEST INCREMENT OF DISP.         1.696E-05   AT NODE      16678   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.923E-07   AT NODE      16257   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  4.941E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.571E-15   AT NODE      11922   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.850E-02   AT NODE      16678   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.967E-03   AT NODE      16296   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.0s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.09       TIME AVG. FORCE       0.997    
 LARGEST SCALED RESIDUAL FORCE     -2.414E-03   AT NODE      37158   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -2.414E-03
 LARGEST INCREMENT OF DISP.         1.695E-05   AT NODE      16678   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.769E-08   AT NODE      14695   DOF  1
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  4.940E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -4.154E-17   AT NODE      42733   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.855E-02   AT NODE      16678   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.184E-03   AT NODE      14859   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -4.855E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.770    
 STEP TIME COMPLETED        512.    ,  TOTAL TIME COMPLETED        7.712E+03


  INCREMENT    78 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  13s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.09       TIME AVG. FORCE       0.998    
 LARGEST SCALED RESIDUAL FORCE     -2.570E-02   AT NODE      37207   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -2.570E-02
 LARGEST INCREMENT OF DISP.         1.697E-05   AT NODE      16678   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.934E-07   AT NODE      16257   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  4.997E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.587E-15   AT NODE      11922   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.768E-02   AT NODE      16679   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.895E-03   AT NODE      16296   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  11s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.09       TIME AVG. FORCE       0.998    
 LARGEST SCALED RESIDUAL FORCE      2.663E-03   AT NODE      39917   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      2.663E-03
 LARGEST INCREMENT OF DISP.         1.696E-05   AT NODE      16678   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.756E-08   AT NODE        168   DOF  3
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  4.996E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -4.276E-17   AT NODE      42733   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.773E-02   AT NODE      16679   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.197E-03   AT NODE      14859   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -4.773E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.780    
 STEP TIME COMPLETED        519.    ,  TOTAL TIME COMPLETED        7.719E+03


  INCREMENT    79 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  14s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.09       TIME AVG. FORCE        1.00    
 LARGEST SCALED RESIDUAL FORCE     -4.441E-02   AT NODE      37155   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -4.441E-02
 LARGEST INCREMENT OF DISP.         1.698E-05   AT NODE      16678   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -1.942E-07   AT NODE      21091   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  5.053E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.606E-15   AT NODE      11921   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.717E-02   AT NODE      16679   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.822E-03   AT NODE      16731   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  12s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.09       TIME AVG. FORCE        1.00    
 LARGEST SCALED RESIDUAL FORCE      2.396E-03   AT NODE      39967   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      2.396E-03
 LARGEST INCREMENT OF DISP.         1.697E-05   AT NODE      16678   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.787E-08   AT NODE        168   DOF  3
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  5.052E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -4.388E-17   AT NODE      42733   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.722E-02   AT NODE      16679   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.206E-03   AT NODE      14859   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -4.722E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.790    
 STEP TIME COMPLETED        525.    ,  TOTAL TIME COMPLETED        7.725E+03


  INCREMENT    80 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.08       TIME AVG. FORCE        1.00    
 LARGEST SCALED RESIDUAL FORCE     -3.865E-02   AT NODE      37204   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -3.865E-02
 LARGEST INCREMENT OF DISP.         1.699E-05   AT NODE      16678   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.948E-07   AT NODE      16251   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  5.111E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.628E-15   AT NODE      11921   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.642E-02   AT NODE      16679   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.758E-03   AT NODE      16731   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.2s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.08       TIME AVG. FORCE        1.00    
 LARGEST SCALED RESIDUAL FORCE      2.463E-03   AT NODE      34859   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      2.463E-03
 LARGEST INCREMENT OF DISP.         1.698E-05   AT NODE      16678   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.807E-08   AT NODE        167   DOF  3
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  5.110E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -4.515E-17   AT NODE      42733   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.647E-02   AT NODE      16679   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.207E-03   AT NODE      14859   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -4.647E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.800    
 STEP TIME COMPLETED        532.    ,  TOTAL TIME COMPLETED        7.732E+03


  INCREMENT    81 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.08       TIME AVG. FORCE        1.00    
 LARGEST SCALED RESIDUAL FORCE     -3.989E-02   AT NODE      40308   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -3.989E-02
 LARGEST INCREMENT OF DISP.         1.700E-05   AT NODE      16679   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.962E-07   AT NODE      16251   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  5.170E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.645E-15   AT NODE      11921   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.547E-02   AT NODE      21117   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.743E-03   AT NODE      16731   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.08       TIME AVG. FORCE        1.00    
 LARGEST SCALED RESIDUAL FORCE     -1.364E-02   AT NODE      39863   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -1.364E-02
 LARGEST INCREMENT OF DISP.         1.699E-05   AT NODE      16679   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.836E-08   AT NODE        167   DOF  3
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  5.169E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -4.617E-17   AT NODE      42733   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.556E-02   AT NODE      21117   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.207E-03   AT NODE      14858   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.08       TIME AVG. FORCE        1.00    
 LARGEST SCALED RESIDUAL FORCE     -1.234E-04   AT NODE      37202   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -1.234E-04
 LARGEST INCREMENT OF DISP.         1.699E-05   AT NODE      16679   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -2.956E-10   AT NODE      39843   DOF  2
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  5.168E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.561E-18   AT NODE      40816   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.557E-02   AT NODE      21117   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.674E-05   AT NODE      14859   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -4.557E-02

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.810    
 STEP TIME COMPLETED        539.    ,  TOTAL TIME COMPLETED        7.739E+03


  INCREMENT    82 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.2s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.08       TIME AVG. FORCE        1.00    
 LARGEST SCALED RESIDUAL FORCE      1.820E-02   AT NODE      34843   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      1.820E-02
 LARGEST INCREMENT OF DISP.         1.701E-05   AT NODE      16679   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -1.967E-07   AT NODE      21092   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  5.229E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.656E-15   AT NODE      11921   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.541E-02   AT NODE      21117   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.990E-03   AT NODE      21129   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.5s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.08       TIME AVG. FORCE        1.00    
 LARGEST SCALED RESIDUAL FORCE     -7.612E-03   AT NODE      39836   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -7.612E-03
 LARGEST INCREMENT OF DISP.         1.701E-05   AT NODE      16679   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.861E-08   AT NODE        167   DOF  3
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  5.227E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -4.714E-17   AT NODE      42733   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.549E-02   AT NODE      21117   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.214E-03   AT NODE      14858   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.0s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       1.08       TIME AVG. FORCE        1.00    
 LARGEST SCALED RESIDUAL FORCE      5.925E-05   AT NODE      32614   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      5.925E-05
 LARGEST INCREMENT OF DISP.         1.701E-05   AT NODE      16679   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -1.619E-10   AT NODE        253   DOF  1
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  5.227E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -2.499E-18   AT NODE      42070   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.549E-02   AT NODE      21117   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        1.778E-05   AT NODE      14859   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -4.549E-02

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.820    
 STEP TIME COMPLETED        545.    ,  TOTAL TIME COMPLETED        7.745E+03


  INCREMENT    83 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.0s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.08       TIME AVG. FORCE        1.00    
 LARGEST SCALED RESIDUAL FORCE     -2.545E-02   AT NODE      37056   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -2.545E-02
 LARGEST INCREMENT OF DISP.         1.702E-05   AT NODE      16679   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.978E-07   AT NODE      16245   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  5.288E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.668E-15   AT NODE      11920   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.494E-02   AT NODE      21117   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.896E-03   AT NODE      21129   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.8s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.08       TIME AVG. FORCE        1.00    
 LARGEST SCALED RESIDUAL FORCE     -8.169E-04   AT NODE      32697   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -8.169E-04
 LARGEST INCREMENT OF DISP.         1.701E-05   AT NODE      16679   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.872E-08   AT NODE        167   DOF  3
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  5.287E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -4.806E-17   AT NODE      42733   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.501E-02   AT NODE      21117   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.213E-03   AT NODE      14858   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -4.501E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.830    
 STEP TIME COMPLETED        552.    ,  TOTAL TIME COMPLETED        7.752E+03


  INCREMENT    84 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.2s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.07       TIME AVG. FORCE        1.00    
 LARGEST SCALED RESIDUAL FORCE     -1.190E-02   AT NODE      39719   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -1.190E-02
 LARGEST INCREMENT OF DISP.         1.703E-05   AT NODE      16625   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.987E-07   AT NODE      16245   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  5.347E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.684E-15   AT NODE      11920   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.402E-02   AT NODE      21117   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.990E-03   AT NODE      21130   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.6s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.07       TIME AVG. FORCE        1.00    
 LARGEST SCALED RESIDUAL FORCE      1.089E-03   AT NODE      39825   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      1.089E-03
 LARGEST INCREMENT OF DISP.         1.702E-05   AT NODE      16625   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.883E-08   AT NODE        166   DOF  3
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  5.346E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -4.902E-17   AT NODE      42733   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.409E-02   AT NODE      21117   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.205E-03   AT NODE      14858   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -4.409E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.840    
 STEP TIME COMPLETED        559.    ,  TOTAL TIME COMPLETED        7.759E+03


  INCREMENT    85 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.2s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.07       TIME AVG. FORCE        1.01    
 LARGEST SCALED RESIDUAL FORCE     -1.287E-02   AT NODE      39919   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -1.287E-02
 LARGEST INCREMENT OF DISP.         1.704E-05   AT NODE      16625   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.991E-07   AT NODE      16245   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  5.407E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.704E-15   AT NODE       3108   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.342E-02   AT NODE      21118   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        3.147E-03   AT NODE      21130   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.0s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.07       TIME AVG. FORCE        1.01    
 LARGEST SCALED RESIDUAL FORCE      1.117E-03   AT NODE      39920   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      1.117E-03
 LARGEST INCREMENT OF DISP.         1.703E-05   AT NODE      16625   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.908E-08   AT NODE        166   DOF  3
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  5.405E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -4.996E-17   AT NODE      42733   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.351E-02   AT NODE      21118   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.200E-03   AT NODE      14857   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -4.351E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.850    
 STEP TIME COMPLETED        565.    ,  TOTAL TIME COMPLETED        7.765E+03


  INCREMENT    86 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.0s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.07       TIME AVG. FORCE        1.01    
 LARGEST SCALED RESIDUAL FORCE     -2.280E-02   AT NODE      36973   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -2.280E-02
 LARGEST INCREMENT OF DISP.         1.705E-05   AT NODE      16625   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -2.002E-07   AT NODE      20814   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  5.466E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.731E-15   AT NODE       3107   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.305E-02   AT NODE      21118   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        3.100E-03   AT NODE      21130   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.2s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.07       TIME AVG. FORCE        1.01    
 LARGEST SCALED RESIDUAL FORCE      3.127E-03   AT NODE      36973   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      3.127E-03
 LARGEST INCREMENT OF DISP.         1.704E-05   AT NODE      16625   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.922E-08   AT NODE        166   DOF  3
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  5.464E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.087E-17   AT NODE      42733   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.313E-02   AT NODE      21118   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.198E-03   AT NODE      14857   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -4.313E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.860    
 STEP TIME COMPLETED        572.    ,  TOTAL TIME COMPLETED        7.772E+03


  INCREMENT    87 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.07       TIME AVG. FORCE        1.01    
 LARGEST SCALED RESIDUAL FORCE     -1.577E-02   AT NODE      37084   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -1.577E-02
 LARGEST INCREMENT OF DISP.         1.706E-05   AT NODE      16626   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -2.005E-07   AT NODE      20814   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  5.524E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.769E-15   AT NODE       3107   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.218E-02   AT NODE      21118   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        3.032E-03   AT NODE      21116   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.07       TIME AVG. FORCE        1.01    
 LARGEST SCALED RESIDUAL FORCE      2.021E-03   AT NODE      36210   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      2.021E-03
 LARGEST INCREMENT OF DISP.         1.705E-05   AT NODE      16626   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.937E-08   AT NODE        166   DOF  3
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  5.523E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.170E-17   AT NODE      42733   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.225E-02   AT NODE      21118   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.187E-03   AT NODE      14857   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -4.225E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.870    
 STEP TIME COMPLETED        579.    ,  TOTAL TIME COMPLETED        7.779E+03


  INCREMENT    88 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.06       TIME AVG. FORCE        1.01    
 LARGEST SCALED RESIDUAL FORCE      3.173E-02   AT NODE      37155   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      3.173E-02
 LARGEST INCREMENT OF DISP.         1.707E-05   AT NODE      16626   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -2.023E-07   AT NODE      21094   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  5.583E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.803E-15   AT NODE       3107   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.150E-02   AT NODE      21119   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        3.136E-03   AT NODE      21116   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.2s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.06       TIME AVG. FORCE        1.01    
 LARGEST SCALED RESIDUAL FORCE      2.271E-03   AT NODE      37070   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      2.271E-03
 LARGEST INCREMENT OF DISP.         1.706E-05   AT NODE      16626   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.951E-08   AT NODE        165   DOF  3
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  5.581E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.241E-17   AT NODE      42733   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.159E-02   AT NODE      21119   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.176E-03   AT NODE      14856   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -4.159E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.880    
 STEP TIME COMPLETED        585.    ,  TOTAL TIME COMPLETED        7.785E+03


  INCREMENT    89 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.0s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.06       TIME AVG. FORCE        1.01    
 LARGEST SCALED RESIDUAL FORCE     -3.662E-02   AT NODE      23109   DOF  1
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -3.662E-02
 LARGEST INCREMENT OF DISP.         1.707E-05   AT NODE      16626   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -2.029E-07   AT NODE      20808   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  5.640E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.832E-15   AT NODE       3107   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.120E-02   AT NODE      21119   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        3.137E-03   AT NODE      21116   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  8.9s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.06       TIME AVG. FORCE        1.01    
 LARGEST SCALED RESIDUAL FORCE     -3.733E-03   AT NODE      37144   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -3.733E-03
 LARGEST INCREMENT OF DISP.         1.707E-05   AT NODE      16626   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.959E-08   AT NODE        165   DOF  3
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  5.639E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.344E-17   AT NODE      42733   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.128E-02   AT NODE      21119   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.172E-03   AT NODE      14856   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -4.128E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.890    
 STEP TIME COMPLETED        592.    ,  TOTAL TIME COMPLETED        7.792E+03


  INCREMENT    90 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.5s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.06       TIME AVG. FORCE        1.01    
 LARGEST SCALED RESIDUAL FORCE     -3.408E-02   AT NODE      37127   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -3.408E-02
 LARGEST INCREMENT OF DISP.         1.708E-05   AT NODE      16626   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -2.034E-07   AT NODE      20808   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  5.697E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.859E-15   AT NODE       3107   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.064E-02   AT NODE      21119   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        3.301E-03   AT NODE      21116   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.3s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.06       TIME AVG. FORCE        1.01    
 LARGEST SCALED RESIDUAL FORCE      3.149E-03   AT NODE      39842   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      3.149E-03
 LARGEST INCREMENT OF DISP.         1.707E-05   AT NODE      16626   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.974E-08   AT NODE        165   DOF  3
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  5.695E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.400E-17   AT NODE      42733   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.071E-02   AT NODE      21119   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.159E-03   AT NODE      14856   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -4.071E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.900    
 STEP TIME COMPLETED        598.    ,  TOTAL TIME COMPLETED        7.798E+03


  INCREMENT    91 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.2s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.06       TIME AVG. FORCE        1.01    
 LARGEST SCALED RESIDUAL FORCE     -3.807E-02   AT NODE      37068   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -3.807E-02
 LARGEST INCREMENT OF DISP.         1.709E-05   AT NODE      16627   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -2.046E-07   AT NODE      20802   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  5.752E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.887E-15   AT NODE       3106   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.039E-02   AT NODE      21120   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        3.542E-03   AT NODE      21116   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.06       TIME AVG. FORCE        1.01    
 LARGEST SCALED RESIDUAL FORCE     -3.300E-03   AT NODE      37143   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -3.300E-03
 LARGEST INCREMENT OF DISP.         1.708E-05   AT NODE      16627   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.975E-08   AT NODE        165   DOF  3
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  5.750E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.445E-17   AT NODE      42733   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.047E-02   AT NODE      21120   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.147E-03   AT NODE      14855   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -4.047E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.910    
 STEP TIME COMPLETED        605.    ,  TOTAL TIME COMPLETED        7.805E+03


  INCREMENT    92 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.05       TIME AVG. FORCE        1.01    
 LARGEST SCALED RESIDUAL FORCE     -5.021E-02   AT NODE      37076   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -5.021E-02
 LARGEST INCREMENT OF DISP.         1.709E-05   AT NODE      16627   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -2.063E-07   AT NODE      20802   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  5.805E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.920E-15   AT NODE       3106   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.013E-02   AT NODE      21120   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        3.361E-03   AT NODE      21117   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  8.9s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.05       TIME AVG. FORCE        1.01    
 LARGEST SCALED RESIDUAL FORCE     -2.987E-03   AT NODE      39815   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -2.987E-03
 LARGEST INCREMENT OF DISP.         1.708E-05   AT NODE      16627   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.980E-08   AT NODE        164   DOF  3
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  5.803E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.495E-17   AT NODE      42810   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.021E-02   AT NODE      21120   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.142E-03   AT NODE      14855   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -4.021E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.920    
 STEP TIME COMPLETED        612.    ,  TOTAL TIME COMPLETED        7.812E+03


  INCREMENT    93 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.2s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.05       TIME AVG. FORCE        1.01    
 LARGEST SCALED RESIDUAL FORCE     -1.462E-02   AT NODE      37067   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -1.462E-02
 LARGEST INCREMENT OF DISP.         1.710E-05   AT NODE      16627   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -2.068E-07   AT NODE      20802   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  5.856E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.948E-15   AT NODE       3106   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.997E-02   AT NODE      21121   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        3.303E-03   AT NODE      20108   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.05       TIME AVG. FORCE        1.01    
 LARGEST SCALED RESIDUAL FORCE      1.562E-03   AT NODE      34921   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      1.562E-03
 LARGEST INCREMENT OF DISP.         1.709E-05   AT NODE      16627   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.988E-08   AT NODE        164   DOF  3
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  5.855E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.563E-17   AT NODE      42733   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.006E-02   AT NODE      21121   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.129E-03   AT NODE      14855   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -4.006E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.930    
 STEP TIME COMPLETED        618.    ,  TOTAL TIME COMPLETED        7.818E+03


  INCREMENT    94 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.2s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.05       TIME AVG. FORCE        1.01    
 LARGEST SCALED RESIDUAL FORCE      9.552E-03   AT NODE      36631   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      9.552E-03
 LARGEST INCREMENT OF DISP.         1.710E-05   AT NODE      16628   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -2.089E-07   AT NODE      20796   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  5.906E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.969E-15   AT NODE       3106   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.000E-02   AT NODE      21121   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        3.078E-03   AT NODE      16626   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  8.8s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.05       TIME AVG. FORCE        1.01    
 LARGEST SCALED RESIDUAL FORCE     -1.610E-03   AT NODE      35008   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -1.610E-03
 LARGEST INCREMENT OF DISP.         1.709E-05   AT NODE      16627   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        1.989E-08   AT NODE        164   DOF  3
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  5.904E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.617E-17   AT NODE      42733   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -4.009E-02   AT NODE      21121   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.119E-03   AT NODE      14854   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -4.009E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.940    
 STEP TIME COMPLETED        625.    ,  TOTAL TIME COMPLETED        7.825E+03


  INCREMENT    95 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.04       TIME AVG. FORCE        1.01    
 LARGEST SCALED RESIDUAL FORCE     -1.168E-02   AT NODE      34899   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -1.168E-02
 LARGEST INCREMENT OF DISP.         1.711E-05   AT NODE      16628   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -2.106E-07   AT NODE      20796   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  5.954E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -1.984E-15   AT NODE       3106   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.991E-02   AT NODE      21121   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.849E-03   AT NODE      21118   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.04       TIME AVG. FORCE        1.01    
 LARGEST SCALED RESIDUAL FORCE      9.151E-04   AT NODE      36141   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      9.151E-04
 LARGEST INCREMENT OF DISP.         1.710E-05   AT NODE      16628   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.000E-08   AT NODE        163   DOF  3
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  5.953E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.663E-17   AT NODE      42733   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.999E-02   AT NODE      21121   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.114E-03   AT NODE      14854   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -3.999E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.950    
 STEP TIME COMPLETED        632.    ,  TOTAL TIME COMPLETED        7.832E+03


  INCREMENT    96 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.0s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.04       TIME AVG. FORCE        1.01    
 LARGEST SCALED RESIDUAL FORCE      6.429E-03   AT NODE        163   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      6.429E-03
 LARGEST INCREMENT OF DISP.         1.711E-05   AT NODE      16628   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -2.112E-07   AT NODE      20796   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  6.001E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -2.008E-15   AT NODE       3105   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.979E-02   AT NODE      21121   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.937E-03   AT NODE      21118   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.04       TIME AVG. FORCE        1.01    
 LARGEST SCALED RESIDUAL FORCE     -8.510E-04   AT NODE      32993   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -8.510E-04
 LARGEST INCREMENT OF DISP.         1.710E-05   AT NODE      16628   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.002E-08   AT NODE        163   DOF  3
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  5.999E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.707E-17   AT NODE      42733   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.986E-02   AT NODE      21121   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.102E-03   AT NODE      14854   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -3.986E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.960    
 STEP TIME COMPLETED        638.    ,  TOTAL TIME COMPLETED        7.838E+03


  INCREMENT    97 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.04       TIME AVG. FORCE        1.01    
 LARGEST SCALED RESIDUAL FORCE      7.138E-03   AT NODE      39729   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      7.138E-03
 LARGEST INCREMENT OF DISP.         1.712E-05   AT NODE      16628   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -2.129E-07   AT NODE      20790   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  6.047E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -2.032E-15   AT NODE       3105   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.982E-02   AT NODE      21122   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.810E-03   AT NODE      21118   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  8.9s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.04       TIME AVG. FORCE        1.01    
 LARGEST SCALED RESIDUAL FORCE     -7.161E-04   AT NODE      18501   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -7.161E-04
 LARGEST INCREMENT OF DISP.         1.711E-05   AT NODE      16628   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.001E-08   AT NODE        163   DOF  3
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  6.045E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.750E-17   AT NODE      42733   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.990E-02   AT NODE      21122   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.094E-03   AT NODE      14853   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -3.990E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.970    
 STEP TIME COMPLETED        645.    ,  TOTAL TIME COMPLETED        7.845E+03


  INCREMENT    98 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  8.9s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.04       TIME AVG. FORCE        1.01    
 LARGEST SCALED RESIDUAL FORCE     -1.610E-02   AT NODE      34920   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -1.610E-02
 LARGEST INCREMENT OF DISP.         1.712E-05   AT NODE      16629   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -2.141E-07   AT NODE      20790   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  6.091E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -2.051E-15   AT NODE       3105   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.977E-02   AT NODE      21122   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.776E-03   AT NODE      21118   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.0s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.04       TIME AVG. FORCE        1.01    
 LARGEST SCALED RESIDUAL FORCE     -1.410E-03   AT NODE      34923   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -1.410E-03
 LARGEST INCREMENT OF DISP.         1.711E-05   AT NODE      16629   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.007E-08   AT NODE        162   DOF  3
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  6.089E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.785E-17   AT NODE      42733   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.985E-02   AT NODE      21122   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.090E-03   AT NODE      14853   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -3.985E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.980    
 STEP TIME COMPLETED        652.    ,  TOTAL TIME COMPLETED        7.852E+03


  INCREMENT    99 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.2s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.03       TIME AVG. FORCE        1.01    
 LARGEST SCALED RESIDUAL FORCE      8.222E-03   AT NODE      39695   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      8.222E-03
 LARGEST INCREMENT OF DISP.         1.713E-05   AT NODE      16629   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -2.142E-07   AT NODE      20790   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  6.134E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -2.066E-15   AT NODE       3105   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.964E-02   AT NODE      21122   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.939E-03   AT NODE      21119   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.03       TIME AVG. FORCE        1.01    
 LARGEST SCALED RESIDUAL FORCE     -8.518E-04   AT NODE      32987   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE     -8.518E-04
 LARGEST INCREMENT OF DISP.         1.712E-05   AT NODE      16629   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.005E-08   AT NODE        162   DOF  3
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  6.133E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.825E-17   AT NODE      42733   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.971E-02   AT NODE      21123   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.077E-03   AT NODE      14853   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -3.971E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED  0.990    
 STEP TIME COMPLETED        658.    ,  TOTAL TIME COMPLETED        7.858E+03


  INCREMENT   100 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT   6.65    
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.3s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.03       TIME AVG. FORCE        1.01    
 LARGEST SCALED RESIDUAL FORCE      6.475E-03   AT NODE        162   DOF  3
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      6.475E-03
 LARGEST INCREMENT OF DISP.         1.713E-05   AT NODE      16629   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.       -2.154E-07   AT NODE      20784   DOF  2
   INSTANCE: CHD2                                                                            
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                  6.176E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -2.078E-15   AT NODE       3104   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.977E-02   AT NODE      21123   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.        2.728E-03   AT NODE      21119   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.
 
	UNSYMMETRIC NEW DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x 1 THREAD PER RANK
        NUMBER OF EQUATIONS:  174160
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      2.375e+11
        SOLVER ELAPSED TIME:  9.1s

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.03       TIME AVG. FORCE        1.01    
 LARGEST SCALED RESIDUAL FORCE      6.323E-04   AT NODE      13745   DOF  2
   INSTANCE: CHD2                                                                            
  CORRESPONDING RESIDUAL FORCE      6.323E-04
 LARGEST INCREMENT OF DISP.         1.712E-05   AT NODE      16629   DOF  2
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO DISP.        2.005E-08   AT NODE        162   DOF  3
   INSTANCE: CHD2                                                                            
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                  6.175E-15   TIME AVG. HEAT FLUX   1.000E-02
 LARGEST RESIDUAL HEAT FLUX        -5.860E-17   AT NODE      42733   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST INCREMENT OF TEMP.        -3.985E-02   AT NODE      21123   DOF 11
   INSTANCE: CHD2                                                                            
 LARGEST CORRECTION TO TEMP.       -2.071E-03   AT NODE      14852   DOF 11
   INSTANCE: CHD2                                                                            
          THERE IS ZERO HEAT FLUX EVERYWHERE
 

 ***WARNING: THERE IS ZERO HEAT FLUX EVERYWHERE IN THE MODEL BASED ON THE 
             DEFAULT CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE HEAT FLUX 
             DURING THE CURRENT ITERATION TO VERIFY THAT THE HEAT FLUX IS SMALL 
             ENOUGH TO BE TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION 
             CONTROLS TO RESET THE CRITERION FOR ZERO HEAT FLUX.

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS  -3.985E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED   6.65    ,  FRACTION OF STEP COMPLETED   1.00    
 STEP TIME COMPLETED        665.    ,  TOTAL TIME COMPLETED        7.865E+03


          THE ANALYSIS HAS BEEN COMPLETED



     ANALYSIS SUMMARY:
     TOTAL OF        110  INCREMENTS
                       0  CUTBACKS IN AUTOMATIC INCREMENTATION
                     233  ITERATIONS INCLUDING CONTACT ITERATIONS IF PRESENT
                     233  PASSES THROUGH THE EQUATION SOLVER OF WHICH 
                     233  INVOLVE MATRIX DECOMPOSITION, INCLUDING
                       0  DECOMPOSITION(S) OF THE MASS MATRIX
                       1  REORDERING OF EQUATIONS TO MINIMIZE WAVEFRONT
                       0  ADDITIONAL RESIDUAL EVALUATIONS FOR LINE SEARCHES
                       0  ADDITIONAL OPERATOR EVALUATIONS FOR LINE SEARCHES
                       3  WARNING MESSAGES DURING USER INPUT PROCESSING
                 2303187  WARNING MESSAGES DURING ANALYSIS
                 2302954  ANALYSIS WARNINGS ARE NUMERICAL PROBLEM MESSAGES
                       0  ANALYSIS WARNINGS ARE NEGATIVE EIGENVALUE MESSAGES
                       0  ERROR MESSAGES



     JOB TIME SUMMARY
       USER TIME (SEC)      =     1.88E+03
       SYSTEM TIME (SEC)    =     1.95E+02
       TOTAL CPU TIME (SEC) =     2.07E+03
       WALLCLOCK TIME (SEC) =         4414
       MEMORY PEAK (GB)     =            1
