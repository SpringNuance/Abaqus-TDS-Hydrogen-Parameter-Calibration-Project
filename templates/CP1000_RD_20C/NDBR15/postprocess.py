# -*- coding: mbcs -*-
#
# Abaqus/Viewer Release 2023.HF4 replay file
# Internal Version: 2023_07_21-20.45.57 RELr425 183702
# Run by nguyenb5 on Sun Jun  9 16:17:16 2024
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=119.81770324707, 
    height=101.190971374512)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()
from viewerModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
import os
os.chdir(os.getcwd())
o2 = session.openOdb(name='geometry.odb')
#: Model: C:/LocalUserData/User-data/nguyenb5/CP1000 processed/CP1000 NDBR15/geometry.odb
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
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.46123, 
    farPlane=0.623297, width=0.0238935, height=0.00859053, 
    viewOffsetX=0.000983696, viewOffsetY=0.0882624)
odb = session.odbs['geometry.odb']
session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('RF', 
    NODAL, ((COMPONENT, 'RF2'), )), ('U', NODAL, ((COMPONENT, 'U2'), )), ), 
    nodePick=(('NDBR15-M-1', 27, (
    '[#f00000 #0:13 #800000 #fc004000 #ff #0:59 #3f8 ]', )), ), )
xy1 = session.xyDataObjects['U:U2 PI: NDBR15-M-1 N: 21']
xy2 = session.xyDataObjects['U:U2 PI: NDBR15-M-1 N: 22']
xy3 = session.xyDataObjects['U:U2 PI: NDBR15-M-1 N: 23']
xy4 = session.xyDataObjects['U:U2 PI: NDBR15-M-1 N: 24']
xy5 = session.xyDataObjects['U:U2 PI: NDBR15-M-1 N: 472']
xy6 = session.xyDataObjects['U:U2 PI: NDBR15-M-1 N: 495']
xy7 = session.xyDataObjects['U:U2 PI: NDBR15-M-1 N: 507']
xy8 = session.xyDataObjects['U:U2 PI: NDBR15-M-1 N: 508']
xy9 = session.xyDataObjects['U:U2 PI: NDBR15-M-1 N: 509']
xy10 = session.xyDataObjects['U:U2 PI: NDBR15-M-1 N: 510']
xy11 = session.xyDataObjects['U:U2 PI: NDBR15-M-1 N: 511']
xy12 = session.xyDataObjects['U:U2 PI: NDBR15-M-1 N: 512']
xy13 = session.xyDataObjects['U:U2 PI: NDBR15-M-1 N: 513']
xy14 = session.xyDataObjects['U:U2 PI: NDBR15-M-1 N: 514']
xy15 = session.xyDataObjects['U:U2 PI: NDBR15-M-1 N: 515']
xy16 = session.xyDataObjects['U:U2 PI: NDBR15-M-1 N: 516']
xy17 = session.xyDataObjects['U:U2 PI: NDBR15-M-1 N: 517']
xy18 = session.xyDataObjects['U:U2 PI: NDBR15-M-1 N: 518']
xy19 = session.xyDataObjects['U:U2 PI: NDBR15-M-1 N: 519']
xy20 = session.xyDataObjects['U:U2 PI: NDBR15-M-1 N: 520']
xy21 = session.xyDataObjects['U:U2 PI: NDBR15-M-1 N: 2436']
xy22 = session.xyDataObjects['U:U2 PI: NDBR15-M-1 N: 2437']
xy23 = session.xyDataObjects['U:U2 PI: NDBR15-M-1 N: 2438']
xy24 = session.xyDataObjects['U:U2 PI: NDBR15-M-1 N: 2439']
xy25 = session.xyDataObjects['U:U2 PI: NDBR15-M-1 N: 2440']
xy26 = session.xyDataObjects['U:U2 PI: NDBR15-M-1 N: 2441']
xy27 = session.xyDataObjects['U:U2 PI: NDBR15-M-1 N: 2442']
xy28 = avg((xy1, xy2, xy3, xy4, xy5, xy6, xy7, xy8, xy9, xy10, xy11, xy12, 
    xy13, xy14, xy15, xy16, xy17, xy18, xy19, xy20, xy21, xy22, xy23, xy24, 
    xy25, xy26, xy27))
xy28.setValues(
    sourceDescription='avg ( ( "U:U2 PI: NDBR15-M-1 N: 21", "U:U2 PI: NDBR15-M-1 N: 22", "U:U2 PI: NDBR15-M-1 N: 23", "U:U2 PI: NDBR15-M-1 N: 24", "U:U2 PI: NDBR15-M-1 N: 472", "U:U2 PI: NDBR15-M-1 N: 495", "U:U2 PI: NDBR15-M-1 N: 507", "U:U2 PI: NDBR15-M-1 N: 508", "U:U2 PI: NDBR15-M-1 N: 509", "U:U2 PI: NDBR15-M-1 N: 510", "U:U2 PI: NDBR15-M-1 N: 511", "U:U2 PI: NDBR15-M-1 N: 512", "U:U2 PI: NDBR15-M-1 N: 513", "U:U2 PI: NDBR15-M-1 N: 514", "U:U2 PI: NDBR15-M-1 N: 515", "U:U2 PI: NDBR15-M-1 N: 516", "U:U2 PI: NDBR15-M-1 N: 517", "U:U2 PI: NDBR15-M-1 N: 518", "U:U2 PI: NDBR15-M-1 N: 519", "U:U2 PI: NDBR15-M-1 N: 520", "U:U2 PI: NDBR15-M-1 N: 2436", "U:U2 PI: NDBR15-M-1 N: 2437", "U:U2 PI: NDBR15-M-1 N: 2438", "U:U2 PI: NDBR15-M-1 N: 2439", "U:U2 PI: NDBR15-M-1 N: 2440", "U:U2 PI: NDBR15-M-1 N: 2441", "U:U2 PI: NDBR15-M-1 N: 2442" ) )')
tmpName = xy28.name
session.xyDataObjects.changeKey(tmpName, 'displacement')
xy1 = session.xyDataObjects['RF:RF2 PI: NDBR15-M-1 N: 21']
xy2 = session.xyDataObjects['RF:RF2 PI: NDBR15-M-1 N: 22']
xy3 = session.xyDataObjects['RF:RF2 PI: NDBR15-M-1 N: 23']
xy4 = session.xyDataObjects['RF:RF2 PI: NDBR15-M-1 N: 24']
xy5 = session.xyDataObjects['RF:RF2 PI: NDBR15-M-1 N: 472']
xy6 = session.xyDataObjects['RF:RF2 PI: NDBR15-M-1 N: 495']
xy7 = session.xyDataObjects['RF:RF2 PI: NDBR15-M-1 N: 507']
xy8 = session.xyDataObjects['RF:RF2 PI: NDBR15-M-1 N: 508']
xy9 = session.xyDataObjects['RF:RF2 PI: NDBR15-M-1 N: 509']
xy10 = session.xyDataObjects['RF:RF2 PI: NDBR15-M-1 N: 510']
xy11 = session.xyDataObjects['RF:RF2 PI: NDBR15-M-1 N: 511']
xy12 = session.xyDataObjects['RF:RF2 PI: NDBR15-M-1 N: 512']
xy13 = session.xyDataObjects['RF:RF2 PI: NDBR15-M-1 N: 513']
xy14 = session.xyDataObjects['RF:RF2 PI: NDBR15-M-1 N: 514']
xy15 = session.xyDataObjects['RF:RF2 PI: NDBR15-M-1 N: 515']
xy16 = session.xyDataObjects['RF:RF2 PI: NDBR15-M-1 N: 516']
xy17 = session.xyDataObjects['RF:RF2 PI: NDBR15-M-1 N: 517']
xy18 = session.xyDataObjects['RF:RF2 PI: NDBR15-M-1 N: 518']
xy19 = session.xyDataObjects['RF:RF2 PI: NDBR15-M-1 N: 519']
xy20 = session.xyDataObjects['RF:RF2 PI: NDBR15-M-1 N: 520']
xy21 = session.xyDataObjects['RF:RF2 PI: NDBR15-M-1 N: 2436']
xy22 = session.xyDataObjects['RF:RF2 PI: NDBR15-M-1 N: 2437']
xy23 = session.xyDataObjects['RF:RF2 PI: NDBR15-M-1 N: 2438']
xy24 = session.xyDataObjects['RF:RF2 PI: NDBR15-M-1 N: 2439']
xy25 = session.xyDataObjects['RF:RF2 PI: NDBR15-M-1 N: 2440']
xy26 = session.xyDataObjects['RF:RF2 PI: NDBR15-M-1 N: 2441']
xy27 = session.xyDataObjects['RF:RF2 PI: NDBR15-M-1 N: 2442']
xy28 = sum((xy1, xy2, xy3, xy4, xy5, xy6, xy7, xy8, xy9, xy10, xy11, xy12, 
    xy13, xy14, xy15, xy16, xy17, xy18, xy19, xy20, xy21, xy22, xy23, xy24, 
    xy25, xy26, xy27))*2
xy28.setValues(
    sourceDescription='sum ( ( "RF:RF2 PI: NDBR15-M-1 N: 21", "RF:RF2 PI: NDBR15-M-1 N: 22", "RF:RF2 PI: NDBR15-M-1 N: 23", "RF:RF2 PI: NDBR15-M-1 N: 24", "RF:RF2 PI: NDBR15-M-1 N: 472", "RF:RF2 PI: NDBR15-M-1 N: 495", "RF:RF2 PI: NDBR15-M-1 N: 507", "RF:RF2 PI: NDBR15-M-1 N: 508", "RF:RF2 PI: NDBR15-M-1 N: 509", "RF:RF2 PI: NDBR15-M-1 N: 510", "RF:RF2 PI: NDBR15-M-1 N: 511", "RF:RF2 PI: NDBR15-M-1 N: 512", "RF:RF2 PI: NDBR15-M-1 N: 513", "RF:RF2 PI: NDBR15-M-1 N: 514", "RF:RF2 PI: NDBR15-M-1 N: 515", "RF:RF2 PI: NDBR15-M-1 N: 516", "RF:RF2 PI: NDBR15-M-1 N: 517", "RF:RF2 PI: NDBR15-M-1 N: 518", "RF:RF2 PI: NDBR15-M-1 N: 519", "RF:RF2 PI: NDBR15-M-1 N: 520", "RF:RF2 PI: NDBR15-M-1 N: 2436", "RF:RF2 PI: NDBR15-M-1 N: 2437", "RF:RF2 PI: NDBR15-M-1 N: 2438", "RF:RF2 PI: NDBR15-M-1 N: 2439", "RF:RF2 PI: NDBR15-M-1 N: 2440", "RF:RF2 PI: NDBR15-M-1 N: 2441", "RF:RF2 PI: NDBR15-M-1 N: 2442" ) ) * 2')
tmpName = xy28.name
session.xyDataObjects.changeKey(tmpName, 'force')
x0 = session.xyDataObjects['displacement']
x1 = session.xyDataObjects['force']
session.xyReportOptions.setValues(numDigits=9)
session.writeXYReport(fileName='FD_curve.txt', appendMode=OFF, xyData=(x0, x1))
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.459764, 
    farPlane=0.624763, width=0.0329635, height=0.0118515, 
    viewOffsetX=0.00418634, viewOffsetY=-0.00894934)
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=100 )