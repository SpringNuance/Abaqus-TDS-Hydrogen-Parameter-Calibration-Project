# -*- coding: mbcs -*-
#
# Abaqus/Viewer Release 2023.HF4 replay file
# Internal Version: 2023_07_21-20.45.57 RELr425 183702
# Run by nguyenb5 on Thu Nov 14 15:37:33 2024
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=103.244789123535, 
    height=113.476852416992)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()
from viewerModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
o2 = session.openOdb(name='CP1000_diffusion.odb')
#: Model: CP1000_diffusion.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       5
#: Number of Node Sets:          5
#: Number of Steps:              1
session.viewports['Viewport: 1'].setValues(displayedObject=o2)
session.viewports['Viewport: 1'].makeCurrent()
odb = session.odbs['CP1000_diffusion.odb']
session.xyDataListFromField(odb=odb, outputPosition=INTEGRATION_POINT, 
    variable=(('SDV10', INTEGRATION_POINT), ('SDV11', INTEGRATION_POINT), (
    'SDV3', INTEGRATION_POINT), ('SDV4', INTEGRATION_POINT), ('SDV5', 
    INTEGRATION_POINT), ('SDV9', INTEGRATION_POINT), ), operator=AVERAGE_ALL, 
    elementSets=(" ALL ELEMENTS", ))
#: Warning: Requested operation will result in the creation of a very large number of xyDataObjects. Performance can be affected. Please reduce the number of specified entities using Display Group operations before re-performing this operation.
x0 = session.xyDataObjects['AVERAGE_SDV3']
x1 = session.xyDataObjects['AVERAGE_SDV4']
x2 = session.xyDataObjects['AVERAGE_SDV5']
x3 = session.xyDataObjects['AVERAGE_SDV9']
x4 = session.xyDataObjects['AVERAGE_SDV10']
x5 = session.xyDataObjects['AVERAGE_SDV11']
session.xyReportOptions.setValues(numDigits=9)
session.writeXYReport(fileName='TDS_measurement.txt', appendMode=OFF, xyData=(
    x0, x1, x2, x3, x4, x5))