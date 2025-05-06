** MATERIALS
** 
*Material, name=Steel
*Density
1.,
*Depvar
      12,
** 1 - H concentration trap 1 (atoms/mm³)
** 2 - H concentration trap 2 (atoms/mm³)
** 3 - H concentration trap 3 (atoms/mm³)
** 4 - total H concentration in traps (atoms/mm³)
** 5 - total H concentration (lattice + traps) (atoms/mm³)
** 6 - H concentration trap 1 (wt ppm)
** 7 - H concentration trap 2 (wt ppm)
** 8 - H concentration trap 3 (wt ppm)
** 9 - total H concentration in traps (wt ppm)
** 10 - total H concentration (lattice + traps) (wt ppm)
** 11 - H concentration in lattice (wt ppm) 
** 12 - u - total H concentration (atoms/mm³)
1, CT1, CT1
2, CT2, CT2
3, CT3, CT3
4, CT, CT
5, C, C
6, CT1_wppm, CT1_wppm
7, CT2_wppm, CT2_wppm
8, CT3_wppm, CT3_wppm
9, CT_wppm, CT_wppm
10, C_wppm, C_wppm
11, CL__wppm,CL__wppm
12, u, u
*User Material, constants=13, type=THERMAL
**Temperature [K], D0 - pre-exponential factor - lattice diffusion [mm²/s], Ea (activation energy for jumping between lattice sites (mJ/mol), xNl (lattice sites, sites/mm³), kflag, Tflag, NoTraps, Wb_1 (mJ/mol), Nt_1 (trap sites/mm³), Wb_2, Nt_2, Wb_3, Nt_3, ..., Wb_n, Nt_n (three traps)
** kflag: 0 - No traps, 1 - With traps
** Tflag: 0 - use temperature specified in material file, 1 - temperature determined through predefined field
 300.0, 0.0723, 6.86E+6, 5.1E+20, 1, 1, 3, 24.9e+06
 5.1e+17, 7.0e+07, 0.0, 2.15e+07,  0.0