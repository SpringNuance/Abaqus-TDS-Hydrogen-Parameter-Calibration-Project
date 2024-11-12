# -*- coding: mbcs -*-
#
# Abaqus/Viewer Release 2023.HF4 replay file
# Internal Version: 2023_07_21-20.45.57 RELr425 183702
# Run by nguyenb5 on Sun Jun  9 16:39:23 2024
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
#: Model: C:/LocalUserData/User-data/nguyenb5/CP1000 processed/CP1000 NDBR6/geometry.odb
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
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.461666, 
    farPlane=0.623036, width=0.0222983, height=0.00801699, 
    viewOffsetX=-0.00137701, viewOffsetY=0.0887664)
odb = session.odbs['geometry.odb']
session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('RF', 
    NODAL, ((COMPONENT, 'RF2'), )), ('U', NODAL, ((COMPONENT, 'U2'), )), ), 
    nodePick=(('NDBR6-M-1', 27, (
    '[#f00000 #0:14 #80000000 #400000 #fffc #0:56 #3f80000 ]', )), ), )
xy1 = session.xyDataObjects['U:U2 PI: NDBR6-M-1 N: 21']
xy2 = session.xyDataObjects['U:U2 PI: NDBR6-M-1 N: 22']
xy3 = session.xyDataObjects['U:U2 PI: NDBR6-M-1 N: 23']
xy4 = session.xyDataObjects['U:U2 PI: NDBR6-M-1 N: 24']
xy5 = session.xyDataObjects['U:U2 PI: NDBR6-M-1 N: 512']
xy6 = session.xyDataObjects['U:U2 PI: NDBR6-M-1 N: 535']
xy7 = session.xyDataObjects['U:U2 PI: NDBR6-M-1 N: 547']
xy8 = session.xyDataObjects['U:U2 PI: NDBR6-M-1 N: 548']
xy9 = session.xyDataObjects['U:U2 PI: NDBR6-M-1 N: 549']
xy10 = session.xyDataObjects['U:U2 PI: NDBR6-M-1 N: 550']
xy11 = session.xyDataObjects['U:U2 PI: NDBR6-M-1 N: 551']
xy12 = session.xyDataObjects['U:U2 PI: NDBR6-M-1 N: 552']
xy13 = session.xyDataObjects['U:U2 PI: NDBR6-M-1 N: 553']
xy14 = session.xyDataObjects['U:U2 PI: NDBR6-M-1 N: 554']
xy15 = session.xyDataObjects['U:U2 PI: NDBR6-M-1 N: 555']
xy16 = session.xyDataObjects['U:U2 PI: NDBR6-M-1 N: 556']
xy17 = session.xyDataObjects['U:U2 PI: NDBR6-M-1 N: 557']
xy18 = session.xyDataObjects['U:U2 PI: NDBR6-M-1 N: 558']
xy19 = session.xyDataObjects['U:U2 PI: NDBR6-M-1 N: 559']
xy20 = session.xyDataObjects['U:U2 PI: NDBR6-M-1 N: 560']
xy21 = session.xyDataObjects['U:U2 PI: NDBR6-M-1 N: 2388']
xy22 = session.xyDataObjects['U:U2 PI: NDBR6-M-1 N: 2389']
xy23 = session.xyDataObjects['U:U2 PI: NDBR6-M-1 N: 2390']
xy24 = session.xyDataObjects['U:U2 PI: NDBR6-M-1 N: 2391']
xy25 = session.xyDataObjects['U:U2 PI: NDBR6-M-1 N: 2392']
xy26 = session.xyDataObjects['U:U2 PI: NDBR6-M-1 N: 2393']
xy27 = session.xyDataObjects['U:U2 PI: NDBR6-M-1 N: 2394']
xy28 = avg((xy1, xy2, xy3, xy4, xy5, xy6, xy7, xy8, xy9, xy10, xy11, xy12, 
    xy13, xy14, xy15, xy16, xy17, xy18, xy19, xy20, xy21, xy22, xy23, xy24, 
    xy25, xy26, xy27))
xy28.setValues(
    sourceDescription='avg ( ( "U:U2 PI: NDBR6-M-1 N: 21", "U:U2 PI: NDBR6-M-1 N: 22", "U:U2 PI: NDBR6-M-1 N: 23", "U:U2 PI: NDBR6-M-1 N: 24", "U:U2 PI: NDBR6-M-1 N: 512", "U:U2 PI: NDBR6-M-1 N: 535", "U:U2 PI: NDBR6-M-1 N: 547", "U:U2 PI: NDBR6-M-1 N: 548", "U:U2 PI: NDBR6-M-1 N: 549", "U:U2 PI: NDBR6-M-1 N: 550", "U:U2 PI: NDBR6-M-1 N: 551", "U:U2 PI: NDBR6-M-1 N: 552", "U:U2 PI: NDBR6-M-1 N: 553", "U:U2 PI: NDBR6-M-1 N: 554", "U:U2 PI: NDBR6-M-1 N: 555", "U:U2 PI: NDBR6-M-1 N: 556", "U:U2 PI: NDBR6-M-1 N: 557", "U:U2 PI: NDBR6-M-1 N: 558", "U:U2 PI: NDBR6-M-1 N: 559", "U:U2 PI: NDBR6-M-1 N: 560", "U:U2 PI: NDBR6-M-1 N: 2388", "U:U2 PI: NDBR6-M-1 N: 2389", "U:U2 PI: NDBR6-M-1 N: 2390", "U:U2 PI: NDBR6-M-1 N: 2391", "U:U2 PI: NDBR6-M-1 N: 2392", "U:U2 PI: NDBR6-M-1 N: 2393", "U:U2 PI: NDBR6-M-1 N: 2394" ) )')
tmpName = xy28.name
session.xyDataObjects.changeKey(tmpName, 'displacement')
xy1 = session.xyDataObjects['RF:RF2 PI: NDBR6-M-1 N: 21']
xy2 = session.xyDataObjects['RF:RF2 PI: NDBR6-M-1 N: 22']
xy3 = session.xyDataObjects['RF:RF2 PI: NDBR6-M-1 N: 23']
xy4 = session.xyDataObjects['RF:RF2 PI: NDBR6-M-1 N: 24']
xy5 = session.xyDataObjects['RF:RF2 PI: NDBR6-M-1 N: 512']
xy6 = session.xyDataObjects['RF:RF2 PI: NDBR6-M-1 N: 535']
xy7 = session.xyDataObjects['RF:RF2 PI: NDBR6-M-1 N: 547']
xy8 = session.xyDataObjects['RF:RF2 PI: NDBR6-M-1 N: 548']
xy9 = session.xyDataObjects['RF:RF2 PI: NDBR6-M-1 N: 549']
xy10 = session.xyDataObjects['RF:RF2 PI: NDBR6-M-1 N: 550']
xy11 = session.xyDataObjects['RF:RF2 PI: NDBR6-M-1 N: 551']
xy12 = session.xyDataObjects['RF:RF2 PI: NDBR6-M-1 N: 552']
xy13 = session.xyDataObjects['RF:RF2 PI: NDBR6-M-1 N: 553']
xy14 = session.xyDataObjects['RF:RF2 PI: NDBR6-M-1 N: 554']
xy15 = session.xyDataObjects['RF:RF2 PI: NDBR6-M-1 N: 555']
xy16 = session.xyDataObjects['RF:RF2 PI: NDBR6-M-1 N: 556']
xy17 = session.xyDataObjects['RF:RF2 PI: NDBR6-M-1 N: 557']
xy18 = session.xyDataObjects['RF:RF2 PI: NDBR6-M-1 N: 558']
xy19 = session.xyDataObjects['RF:RF2 PI: NDBR6-M-1 N: 559']
xy20 = session.xyDataObjects['RF:RF2 PI: NDBR6-M-1 N: 560']
xy21 = session.xyDataObjects['RF:RF2 PI: NDBR6-M-1 N: 2388']
xy22 = session.xyDataObjects['RF:RF2 PI: NDBR6-M-1 N: 2389']
xy23 = session.xyDataObjects['RF:RF2 PI: NDBR6-M-1 N: 2390']
xy24 = session.xyDataObjects['RF:RF2 PI: NDBR6-M-1 N: 2391']
xy25 = session.xyDataObjects['RF:RF2 PI: NDBR6-M-1 N: 2392']
xy26 = session.xyDataObjects['RF:RF2 PI: NDBR6-M-1 N: 2393']
xy27 = session.xyDataObjects['RF:RF2 PI: NDBR6-M-1 N: 2394']
xy28 = sum((xy1, xy2, xy3, xy4, xy5, xy6, xy7, xy8, xy9, xy10, xy11, xy12, 
    xy13, xy14, xy15, xy16, xy17, xy18, xy19, xy20, xy21, xy22, xy23, xy24, 
    xy25, xy26, xy27))*2
xy28.setValues(
    sourceDescription='sum ( ( "RF:RF2 PI: NDBR6-M-1 N: 21", "RF:RF2 PI: NDBR6-M-1 N: 22", "RF:RF2 PI: NDBR6-M-1 N: 23", "RF:RF2 PI: NDBR6-M-1 N: 24", "RF:RF2 PI: NDBR6-M-1 N: 512", "RF:RF2 PI: NDBR6-M-1 N: 535", "RF:RF2 PI: NDBR6-M-1 N: 547", "RF:RF2 PI: NDBR6-M-1 N: 548", "RF:RF2 PI: NDBR6-M-1 N: 549", "RF:RF2 PI: NDBR6-M-1 N: 550", "RF:RF2 PI: NDBR6-M-1 N: 551", "RF:RF2 PI: NDBR6-M-1 N: 552", "RF:RF2 PI: NDBR6-M-1 N: 553", "RF:RF2 PI: NDBR6-M-1 N: 554", "RF:RF2 PI: NDBR6-M-1 N: 555", "RF:RF2 PI: NDBR6-M-1 N: 556", "RF:RF2 PI: NDBR6-M-1 N: 557", "RF:RF2 PI: NDBR6-M-1 N: 558", "RF:RF2 PI: NDBR6-M-1 N: 559", "RF:RF2 PI: NDBR6-M-1 N: 560", "RF:RF2 PI: NDBR6-M-1 N: 2388", "RF:RF2 PI: NDBR6-M-1 N: 2389", "RF:RF2 PI: NDBR6-M-1 N: 2390", "RF:RF2 PI: NDBR6-M-1 N: 2391", "RF:RF2 PI: NDBR6-M-1 N: 2392", "RF:RF2 PI: NDBR6-M-1 N: 2393", "RF:RF2 PI: NDBR6-M-1 N: 2394" ) ) * 2')
tmpName = xy28.name
session.xyDataObjects.changeKey(tmpName, 'force')
x0 = session.xyDataObjects['displacement']
x1 = session.xyDataObjects['force']
session.xyReportOptions.setValues(numDigits=9)
session.writeXYReport(fileName='FD_curve.txt', appendMode=OFF, xyData=(x0, x1))