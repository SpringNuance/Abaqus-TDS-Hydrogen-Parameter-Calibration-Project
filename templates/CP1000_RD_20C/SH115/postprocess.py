# -*- coding: mbcs -*-
#
# Abaqus/Viewer Release 2023.HF4 replay file
# Internal Version: 2023_07_21-20.45.57 RELr425 183702
# Run by nguyenb5 on Wed Jun 12 15:58:25 2024
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=119.81770324707, 
    height=113.030090332031)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()
from viewerModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
import os
os.chdir(os.getcwd())
o2 = session.openOdb(name='geometry.odb')
#: Model: C:/LocalUserData/User-data/nguyenb5/CP1000 plastic (Abaqus solver isotropic)/CP1000 SH115/geometry.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       7
#: Number of Node Sets:          9
#: Number of Steps:              1
session.viewports['Viewport: 1'].setValues(displayedObject=o2)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.460857, 
    farPlane=0.623428, width=0.0237367, height=0.00963534, 
    viewOffsetX=-0.00222663, viewOffsetY=0.0890747)
odb = session.odbs['geometry.odb']
session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('RF', 
    NODAL, ((COMPONENT, 'RF2'), )), ('U', NODAL, ((COMPONENT, 'U2'), )), ), 
    nodePick=(('SH115-M-1', 27, ('[#0 #70 #3800 #1c0000 #e000000 #0 #7', 
    ' #380 #1c000 #e00000 #70000000 ]', )), ), )
xy1 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 37']
xy2 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 38']
xy3 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 39']
xy4 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 76']
xy5 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 77']
xy6 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 78']
xy7 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 115']
xy8 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 116']
xy9 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 117']
xy10 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 154']
xy11 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 155']
xy12 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 156']
xy13 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 193']
xy14 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 194']
xy15 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 195']
xy16 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 232']
xy17 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 233']
xy18 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 234']
xy19 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 271']
xy20 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 272']
xy21 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 273']
xy22 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 310']
xy23 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 311']
xy24 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 312']
xy25 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 349']
xy26 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 350']
xy27 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 351']
xy28 = avg((xy1, xy2, xy3, xy4, xy5, xy6, xy7, xy8, xy9, xy10, xy11, xy12, 
    xy13, xy14, xy15, xy16, xy17, xy18, xy19, xy20, xy21, xy22, xy23, xy24, 
    xy25, xy26, xy27))
xy28.setValues(
    sourceDescription='avg ( ( "U:U2 PI: SH115-M-1 N: 37", "U:U2 PI: SH115-M-1 N: 38", "U:U2 PI: SH115-M-1 N: 39", "U:U2 PI: SH115-M-1 N: 76", "U:U2 PI: SH115-M-1 N: 77", "U:U2 PI: SH115-M-1 N: 78", "U:U2 PI: SH115-M-1 N: 115", "U:U2 PI: SH115-M-1 N: 116", "U:U2 PI: SH115-M-1 N: 117", "U:U2 PI: SH115-M-1 N: 154", "U:U2 PI: SH115-M-1 N: 155", "U:U2 PI: SH115-M-1 N: 156", "U:U2 PI: SH115-M-1 N: 193", "U:U2 PI: SH115-M-1 N: 194", "U:U2 PI: SH115-M-1 N: 195", "U:U2 PI: SH115-M-1 N: 232", "U:U2 PI: SH115-M-1 N: 233", "U:U2 PI: SH115-M-1 N: 234", "U:U2 PI: SH115-M-1 N: 271", "U:U2 PI: SH115-M-1 N: 272", "U:U2 PI: SH115-M-1 N: 273", "U:U2 PI: SH115-M-1 N: 310", "U:U2 PI: SH115-M-1 N: 311", "U:U2 PI: SH115-M-1 N: 312", "U:U2 PI: SH115-M-1 N: 349", "U:U2 PI: SH115-M-1 N: 350", "U:U2 PI: SH115-M-1 N: 351" ) ) ')
tmpName = xy28.name
session.xyDataObjects.changeKey(tmpName, 'displacement')
xy1 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 37']
xy2 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 38']
xy3 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 39']
xy4 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 76']
xy5 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 77']
xy6 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 78']
xy7 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 115']
xy8 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 116']
xy9 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 117']
xy10 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 154']
xy11 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 155']
xy12 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 156']
xy13 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 193']
xy14 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 194']
xy15 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 195']
xy16 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 232']
xy17 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 233']
xy18 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 234']
xy19 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 271']
xy20 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 272']
xy21 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 273']
xy22 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 310']
xy23 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 311']
xy24 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 312']
xy25 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 349']
xy26 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 350']
xy27 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 351']
xy28 = sum((xy1, xy2, xy3, xy4, xy5, xy6, xy7, xy8, xy9, xy10, xy11, xy12, 
    xy13, xy14, xy15, xy16, xy17, xy18, xy19, xy20, xy21, xy22, xy23, xy24, 
    xy25, xy26, xy27))*2
xy28.setValues(
    sourceDescription='sum ( ( "RF:RF2 PI: SH115-M-1 N: 37", "RF:RF2 PI: SH115-M-1 N: 38", "RF:RF2 PI: SH115-M-1 N: 39", "RF:RF2 PI: SH115-M-1 N: 76", "RF:RF2 PI: SH115-M-1 N: 77", "RF:RF2 PI: SH115-M-1 N: 78", "RF:RF2 PI: SH115-M-1 N: 115", "RF:RF2 PI: SH115-M-1 N: 116", "RF:RF2 PI: SH115-M-1 N: 117", "RF:RF2 PI: SH115-M-1 N: 154", "RF:RF2 PI: SH115-M-1 N: 155", "RF:RF2 PI: SH115-M-1 N: 156", "RF:RF2 PI: SH115-M-1 N: 193", "RF:RF2 PI: SH115-M-1 N: 194", "RF:RF2 PI: SH115-M-1 N: 195", "RF:RF2 PI: SH115-M-1 N: 232", "RF:RF2 PI: SH115-M-1 N: 233", "RF:RF2 PI: SH115-M-1 N: 234", "RF:RF2 PI: SH115-M-1 N: 271", "RF:RF2 PI: SH115-M-1 N: 272", "RF:RF2 PI: SH115-M-1 N: 273", "RF:RF2 PI: SH115-M-1 N: 310", "RF:RF2 PI: SH115-M-1 N: 311", "RF:RF2 PI: SH115-M-1 N: 312", "RF:RF2 PI: SH115-M-1 N: 349", "RF:RF2 PI: SH115-M-1 N: 350", "RF:RF2 PI: SH115-M-1 N: 351" ) ) * 2')
tmpName = xy28.name
session.xyDataObjects.changeKey(tmpName, 'force')
x0 = session.xyDataObjects['displacement']
x1 = session.xyDataObjects['force']
session.xyReportOptions.setValues(numDigits=9)
session.writeXYReport(fileName='FD_curve.txt', appendMode=OFF, xyData=(x0, x1))