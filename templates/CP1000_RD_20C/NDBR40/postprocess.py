# -*- coding: mbcs -*-
#
# Abaqus/Viewer Release 2023.HF4 replay file
# Internal Version: 2023_07_21-20.45.57 RELr425 183702
# Run by nguyenb5 on Sun Jun  9 15:55:40 2024
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
#: Model: C:/LocalUserData/User-data/nguyenb5/CP1000 processed/CP1000 NDBR40/geometry.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       8
#: Number of Node Sets:          10
#: Number of Steps:              1
session.viewports['Viewport: 1'].setValues(displayedObject=o2)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.461381, 
    farPlane=0.622953, width=0.0209475, height=0.00753132, 
    viewOffsetX=3.2995e-06, viewOffsetY=-0.00991503)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.460545, 
    farPlane=0.623789, width=0.0261174, height=0.00939009, 
    viewOffsetX=-0.000908767, viewOffsetY=0.088889)
odb = session.odbs['geometry.odb']
session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('RF', 
    NODAL, ((COMPONENT, 'RF2'), )), ('U', NODAL, ((COMPONENT, 'U2'), )), ), 
    nodePick=(('NDBR40-M-1', 27, (
    '[#f00000 #0:13 #80000000 #400000 #fffc #0:56 #3f8 ]', )), ), )
xy1 = session.xyDataObjects['U:U2 PI: NDBR40-M-1 N: 21']
xy2 = session.xyDataObjects['U:U2 PI: NDBR40-M-1 N: 22']
xy3 = session.xyDataObjects['U:U2 PI: NDBR40-M-1 N: 23']
xy4 = session.xyDataObjects['U:U2 PI: NDBR40-M-1 N: 24']
xy5 = session.xyDataObjects['U:U2 PI: NDBR40-M-1 N: 480']
xy6 = session.xyDataObjects['U:U2 PI: NDBR40-M-1 N: 503']
xy7 = session.xyDataObjects['U:U2 PI: NDBR40-M-1 N: 515']
xy8 = session.xyDataObjects['U:U2 PI: NDBR40-M-1 N: 516']
xy9 = session.xyDataObjects['U:U2 PI: NDBR40-M-1 N: 517']
xy10 = session.xyDataObjects['U:U2 PI: NDBR40-M-1 N: 518']
xy11 = session.xyDataObjects['U:U2 PI: NDBR40-M-1 N: 519']
xy12 = session.xyDataObjects['U:U2 PI: NDBR40-M-1 N: 520']
xy13 = session.xyDataObjects['U:U2 PI: NDBR40-M-1 N: 521']
xy14 = session.xyDataObjects['U:U2 PI: NDBR40-M-1 N: 522']
xy15 = session.xyDataObjects['U:U2 PI: NDBR40-M-1 N: 523']
xy16 = session.xyDataObjects['U:U2 PI: NDBR40-M-1 N: 524']
xy17 = session.xyDataObjects['U:U2 PI: NDBR40-M-1 N: 525']
xy18 = session.xyDataObjects['U:U2 PI: NDBR40-M-1 N: 526']
xy19 = session.xyDataObjects['U:U2 PI: NDBR40-M-1 N: 527']
xy20 = session.xyDataObjects['U:U2 PI: NDBR40-M-1 N: 528']
xy21 = session.xyDataObjects['U:U2 PI: NDBR40-M-1 N: 2340']
xy22 = session.xyDataObjects['U:U2 PI: NDBR40-M-1 N: 2341']
xy23 = session.xyDataObjects['U:U2 PI: NDBR40-M-1 N: 2342']
xy24 = session.xyDataObjects['U:U2 PI: NDBR40-M-1 N: 2343']
xy25 = session.xyDataObjects['U:U2 PI: NDBR40-M-1 N: 2344']
xy26 = session.xyDataObjects['U:U2 PI: NDBR40-M-1 N: 2345']
xy27 = session.xyDataObjects['U:U2 PI: NDBR40-M-1 N: 2346']
xy28 = avg((xy1, xy2, xy3, xy4, xy5, xy6, xy7, xy8, xy9, xy10, xy11, xy12, 
    xy13, xy14, xy15, xy16, xy17, xy18, xy19, xy20, xy21, xy22, xy23, xy24, 
    xy25, xy26, xy27))
xy28.setValues(
    sourceDescription='avg ( ( "U:U2 PI: NDBR40-M-1 N: 21", "U:U2 PI: NDBR40-M-1 N: 22", "U:U2 PI: NDBR40-M-1 N: 23", "U:U2 PI: NDBR40-M-1 N: 24", "U:U2 PI: NDBR40-M-1 N: 480", "U:U2 PI: NDBR40-M-1 N: 503", "U:U2 PI: NDBR40-M-1 N: 515", "U:U2 PI: NDBR40-M-1 N: 516", "U:U2 PI: NDBR40-M-1 N: 517", "U:U2 PI: NDBR40-M-1 N: 518", "U:U2 PI: NDBR40-M-1 N: 519", "U:U2 PI: NDBR40-M-1 N: 520", "U:U2 PI: NDBR40-M-1 N: 521", "U:U2 PI: NDBR40-M-1 N: 522", "U:U2 PI: NDBR40-M-1 N: 523", "U:U2 PI: NDBR40-M-1 N: 524", "U:U2 PI: NDBR40-M-1 N: 525", "U:U2 PI: NDBR40-M-1 N: 526", "U:U2 PI: NDBR40-M-1 N: 527", "U:U2 PI: NDBR40-M-1 N: 528", "U:U2 PI: NDBR40-M-1 N: 2340", "U:U2 PI: NDBR40-M-1 N: 2341", "U:U2 PI: NDBR40-M-1 N: 2342", "U:U2 PI: NDBR40-M-1 N: 2343", "U:U2 PI: NDBR40-M-1 N: 2344", "U:U2 PI: NDBR40-M-1 N: 2345", "U:U2 PI: NDBR40-M-1 N: 2346" ) )')
tmpName = xy28.name
session.xyDataObjects.changeKey(tmpName, 'displacement')
xy1 = session.xyDataObjects['RF:RF2 PI: NDBR40-M-1 N: 21']
xy2 = session.xyDataObjects['RF:RF2 PI: NDBR40-M-1 N: 22']
xy3 = session.xyDataObjects['RF:RF2 PI: NDBR40-M-1 N: 23']
xy4 = session.xyDataObjects['RF:RF2 PI: NDBR40-M-1 N: 24']
xy5 = session.xyDataObjects['RF:RF2 PI: NDBR40-M-1 N: 480']
xy6 = session.xyDataObjects['RF:RF2 PI: NDBR40-M-1 N: 503']
xy7 = session.xyDataObjects['RF:RF2 PI: NDBR40-M-1 N: 515']
xy8 = session.xyDataObjects['RF:RF2 PI: NDBR40-M-1 N: 516']
xy9 = session.xyDataObjects['RF:RF2 PI: NDBR40-M-1 N: 517']
xy10 = session.xyDataObjects['RF:RF2 PI: NDBR40-M-1 N: 518']
xy11 = session.xyDataObjects['RF:RF2 PI: NDBR40-M-1 N: 519']
xy12 = session.xyDataObjects['RF:RF2 PI: NDBR40-M-1 N: 520']
xy13 = session.xyDataObjects['RF:RF2 PI: NDBR40-M-1 N: 521']
xy14 = session.xyDataObjects['RF:RF2 PI: NDBR40-M-1 N: 522']
xy15 = session.xyDataObjects['RF:RF2 PI: NDBR40-M-1 N: 523']
xy16 = session.xyDataObjects['RF:RF2 PI: NDBR40-M-1 N: 524']
xy17 = session.xyDataObjects['RF:RF2 PI: NDBR40-M-1 N: 525']
xy18 = session.xyDataObjects['RF:RF2 PI: NDBR40-M-1 N: 526']
xy19 = session.xyDataObjects['RF:RF2 PI: NDBR40-M-1 N: 527']
xy20 = session.xyDataObjects['RF:RF2 PI: NDBR40-M-1 N: 528']
xy21 = session.xyDataObjects['RF:RF2 PI: NDBR40-M-1 N: 2340']
xy22 = session.xyDataObjects['RF:RF2 PI: NDBR40-M-1 N: 2341']
xy23 = session.xyDataObjects['RF:RF2 PI: NDBR40-M-1 N: 2342']
xy24 = session.xyDataObjects['RF:RF2 PI: NDBR40-M-1 N: 2343']
xy25 = session.xyDataObjects['RF:RF2 PI: NDBR40-M-1 N: 2344']
xy26 = session.xyDataObjects['RF:RF2 PI: NDBR40-M-1 N: 2345']
xy27 = session.xyDataObjects['RF:RF2 PI: NDBR40-M-1 N: 2346']
xy28 = sum((xy1, xy2, xy3, xy4, xy5, xy6, xy7, xy8, xy9, xy10, xy11, xy12, 
    xy13, xy14, xy15, xy16, xy17, xy18, xy19, xy20, xy21, xy22, xy23, xy24, 
    xy25, xy26, xy27))*2
xy28.setValues(
    sourceDescription='sum ( ( "RF:RF2 PI: NDBR40-M-1 N: 21", "RF:RF2 PI: NDBR40-M-1 N: 22", "RF:RF2 PI: NDBR40-M-1 N: 23", "RF:RF2 PI: NDBR40-M-1 N: 24", "RF:RF2 PI: NDBR40-M-1 N: 480", "RF:RF2 PI: NDBR40-M-1 N: 503", "RF:RF2 PI: NDBR40-M-1 N: 515", "RF:RF2 PI: NDBR40-M-1 N: 516", "RF:RF2 PI: NDBR40-M-1 N: 517", "RF:RF2 PI: NDBR40-M-1 N: 518", "RF:RF2 PI: NDBR40-M-1 N: 519", "RF:RF2 PI: NDBR40-M-1 N: 520", "RF:RF2 PI: NDBR40-M-1 N: 521", "RF:RF2 PI: NDBR40-M-1 N: 522", "RF:RF2 PI: NDBR40-M-1 N: 523", "RF:RF2 PI: NDBR40-M-1 N: 524", "RF:RF2 PI: NDBR40-M-1 N: 525", "RF:RF2 PI: NDBR40-M-1 N: 526", "RF:RF2 PI: NDBR40-M-1 N: 527", "RF:RF2 PI: NDBR40-M-1 N: 528", "RF:RF2 PI: NDBR40-M-1 N: 2340", "RF:RF2 PI: NDBR40-M-1 N: 2341", "RF:RF2 PI: NDBR40-M-1 N: 2342", "RF:RF2 PI: NDBR40-M-1 N: 2343", "RF:RF2 PI: NDBR40-M-1 N: 2344", "RF:RF2 PI: NDBR40-M-1 N: 2345", "RF:RF2 PI: NDBR40-M-1 N: 2346" ) ) * 2')
tmpName = xy28.name
session.xyDataObjects.changeKey(tmpName, 'force')
x0 = session.xyDataObjects['displacement']
x1 = session.xyDataObjects['force']
session.xyReportOptions.setValues(numDigits=9)
session.writeXYReport(fileName='FD_curve.txt', appendMode=OFF, xyData=(x0, x1))