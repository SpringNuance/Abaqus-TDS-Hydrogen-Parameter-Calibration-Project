# -*- coding: mbcs -*-
#
# Abaqus/Viewer Release 2023.HF4 replay file
# Internal Version: 2023_07_21-20.45.57 RELr425 183702
# Run by nguyenb5 on Wed Aug 28 12:56:58 2024
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=90.2552032470703, 
    height=144.303237915039)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()
from viewerModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
import os
os.chdir(os.getcwd())
o2 = session.openOdb(name='SupraPlus_4420_diffusion.odb')
#: Model: C:/LocalUserData/User-data/nguyenb5/hydrogen_diffusion_BC/SupraPlus_4420_steel/SupraPlus_4420_diffusion.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       4
#: Number of Node Sets:          4
#: Number of Steps:              1
session.viewports['Viewport: 1'].setValues(displayedObject=o2)
session.viewports['Viewport: 1'].makeCurrent()
odbName=session.viewports[session.currentViewportName].odbDisplay.name
session.odbData[odbName].setValues(activeFrames=(('Step-1', (200, )), ))
odb = session.odbs['SupraPlus_4420_diffusion.odb']
session.xyDataListFromField(odb=odb, outputPosition=INTEGRATION_POINT, 
    variable=(('SDV1', INTEGRATION_POINT), ('SDV2', INTEGRATION_POINT), ), 
    operator=AVERAGE_ALL, elementSets=(" ALL ELEMENTS", ))
#: Warning: Requested operation will result in the creation of a very large number of xyDataObjects. Performance can be affected. Please reduce the number of specified entities using Display Group operations before re-performing this operation.
x0 = session.xyDataObjects['AVERAGE_SDV1']
x1 = session.xyDataObjects['AVERAGE_SDV2']
session.xyReportOptions.setValues(numDigits=9)
session.writeXYReport(fileName='concentration.txt', appendMode=OFF, xyData=(x0, 
    x1))
