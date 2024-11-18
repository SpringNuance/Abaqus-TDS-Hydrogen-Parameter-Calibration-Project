# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 2023.HF4 replay file
# Internal Version: 2023_07_21-20.45.57 RELr425 183702
# Run by nguyenb5 on Mon Nov 18 20:12:33 2024
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(1.11979, 1.1169), width=164.833, 
    height=110.796)
session.viewports['Viewport: 1'].makeCurrent()
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
execfile('postprocess.py', __main__.__dict__)
#: Model: C:/LocalUserData/User-data/nguyenb5/Abaqus-TDS-Hydrogen-Bayesian-Optimization/sims_iteration/CP1000_RD_20C/iteration_97/prediction_1/CP1000_diffusion.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       5
#: Number of Node Sets:          5
#: Number of Steps:              1
#: Warning: Requested operation will result in the creation of a very large number of xyDataObjects. Performance can be affected. Please reduce the number of specified entities using Display Group operations before re-performing this operation.
print 'RT script done'
#: RT script done
