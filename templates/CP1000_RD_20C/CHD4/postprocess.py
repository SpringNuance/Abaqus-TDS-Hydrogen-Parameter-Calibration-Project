# -*- coding: mbcs -*-
#
# Abaqus/Viewer Release 2023.HF4 replay file
# Internal Version: 2023_07_21-20.45.57 RELr425 183702
# Run by nguyenb5 on Sun Jun  9 17:29:32 2024
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=119.81770324707, 
    height=124.422454833984)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()
from viewerModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
import os
os.chdir(os.getcwd())
o2 = session.openOdb(name='geometry.odb')
#: Model: C:/LocalUserData/User-data/nguyenb5/CP1000 processed/CP1000 CHD4/geometry.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       11
#: Number of Node Sets:          13
#: Number of Steps:              1
session.viewports['Viewport: 1'].setValues(displayedObject=o2)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.462885, 
    farPlane=0.62268, width=0.0201122, height=0.00909706, 
    viewOffsetX=-0.00125127, viewOffsetY=0.087813)
odb = session.odbs['geometry.odb']
session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('RF', 
    NODAL, ((COMPONENT, 'RF2'), )), ('U', NODAL, ((COMPONENT, 'U2'), )), ), 
    nodePick=(('CHD4-M-1', 27, (
    '[#6c000 #0:10 #10000010 #1fc07f00 #0:91 #7f000000 ]', )), ), )
xy1 = session.xyDataObjects['U:U2 PI: CHD4-M-1 N: 15']
xy2 = session.xyDataObjects['U:U2 PI: CHD4-M-1 N: 16']
xy3 = session.xyDataObjects['U:U2 PI: CHD4-M-1 N: 18']
xy4 = session.xyDataObjects['U:U2 PI: CHD4-M-1 N: 19']
xy5 = session.xyDataObjects['U:U2 PI: CHD4-M-1 N: 357']
xy6 = session.xyDataObjects['U:U2 PI: CHD4-M-1 N: 381']
xy7 = session.xyDataObjects['U:U2 PI: CHD4-M-1 N: 393']
xy8 = session.xyDataObjects['U:U2 PI: CHD4-M-1 N: 394']
xy9 = session.xyDataObjects['U:U2 PI: CHD4-M-1 N: 395']
xy10 = session.xyDataObjects['U:U2 PI: CHD4-M-1 N: 396']
xy11 = session.xyDataObjects['U:U2 PI: CHD4-M-1 N: 397']
xy12 = session.xyDataObjects['U:U2 PI: CHD4-M-1 N: 398']
xy13 = session.xyDataObjects['U:U2 PI: CHD4-M-1 N: 399']
xy14 = session.xyDataObjects['U:U2 PI: CHD4-M-1 N: 407']
xy15 = session.xyDataObjects['U:U2 PI: CHD4-M-1 N: 408']
xy16 = session.xyDataObjects['U:U2 PI: CHD4-M-1 N: 409']
xy17 = session.xyDataObjects['U:U2 PI: CHD4-M-1 N: 410']
xy18 = session.xyDataObjects['U:U2 PI: CHD4-M-1 N: 411']
xy19 = session.xyDataObjects['U:U2 PI: CHD4-M-1 N: 412']
xy20 = session.xyDataObjects['U:U2 PI: CHD4-M-1 N: 413']
xy21 = session.xyDataObjects['U:U2 PI: CHD4-M-1 N: 3353']
xy22 = session.xyDataObjects['U:U2 PI: CHD4-M-1 N: 3354']
xy23 = session.xyDataObjects['U:U2 PI: CHD4-M-1 N: 3355']
xy24 = session.xyDataObjects['U:U2 PI: CHD4-M-1 N: 3356']
xy25 = session.xyDataObjects['U:U2 PI: CHD4-M-1 N: 3357']
xy26 = session.xyDataObjects['U:U2 PI: CHD4-M-1 N: 3358']
xy27 = session.xyDataObjects['U:U2 PI: CHD4-M-1 N: 3359']
xy28 = avg((xy1, xy2, xy3, xy4, xy5, xy6, xy7, xy8, xy9, xy10, xy11, xy12, 
    xy13, xy14, xy15, xy16, xy17, xy18, xy19, xy20, xy21, xy22, xy23, xy24, 
    xy25, xy26, xy27))
xy28.setValues(
    sourceDescription='avg ( ( "U:U2 PI: CHD4-M-1 N: 15", "U:U2 PI: CHD4-M-1 N: 16", "U:U2 PI: CHD4-M-1 N: 18", "U:U2 PI: CHD4-M-1 N: 19", "U:U2 PI: CHD4-M-1 N: 357", "U:U2 PI: CHD4-M-1 N: 381", "U:U2 PI: CHD4-M-1 N: 393", "U:U2 PI: CHD4-M-1 N: 394", "U:U2 PI: CHD4-M-1 N: 395", "U:U2 PI: CHD4-M-1 N: 396", "U:U2 PI: CHD4-M-1 N: 397", "U:U2 PI: CHD4-M-1 N: 398", "U:U2 PI: CHD4-M-1 N: 399", "U:U2 PI: CHD4-M-1 N: 407", "U:U2 PI: CHD4-M-1 N: 408", "U:U2 PI: CHD4-M-1 N: 409", "U:U2 PI: CHD4-M-1 N: 410", "U:U2 PI: CHD4-M-1 N: 411", "U:U2 PI: CHD4-M-1 N: 412", "U:U2 PI: CHD4-M-1 N: 413", "U:U2 PI: CHD4-M-1 N: 3353", "U:U2 PI: CHD4-M-1 N: 3354", "U:U2 PI: CHD4-M-1 N: 3355", "U:U2 PI: CHD4-M-1 N: 3356", "U:U2 PI: CHD4-M-1 N: 3357", "U:U2 PI: CHD4-M-1 N: 3358", "U:U2 PI: CHD4-M-1 N: 3359" ) )')
tmpName = xy28.name
session.xyDataObjects.changeKey(tmpName, 'displacement')
xy1 = session.xyDataObjects['RF:RF2 PI: CHD4-M-1 N: 15']
xy2 = session.xyDataObjects['RF:RF2 PI: CHD4-M-1 N: 16']
xy3 = session.xyDataObjects['RF:RF2 PI: CHD4-M-1 N: 18']
xy4 = session.xyDataObjects['RF:RF2 PI: CHD4-M-1 N: 19']
xy5 = session.xyDataObjects['RF:RF2 PI: CHD4-M-1 N: 357']
xy6 = session.xyDataObjects['RF:RF2 PI: CHD4-M-1 N: 381']
xy7 = session.xyDataObjects['RF:RF2 PI: CHD4-M-1 N: 393']
xy8 = session.xyDataObjects['RF:RF2 PI: CHD4-M-1 N: 394']
xy9 = session.xyDataObjects['RF:RF2 PI: CHD4-M-1 N: 395']
xy10 = session.xyDataObjects['RF:RF2 PI: CHD4-M-1 N: 396']
xy11 = session.xyDataObjects['RF:RF2 PI: CHD4-M-1 N: 397']
xy12 = session.xyDataObjects['RF:RF2 PI: CHD4-M-1 N: 398']
xy13 = session.xyDataObjects['RF:RF2 PI: CHD4-M-1 N: 399']
xy14 = session.xyDataObjects['RF:RF2 PI: CHD4-M-1 N: 407']
xy15 = session.xyDataObjects['RF:RF2 PI: CHD4-M-1 N: 408']
xy16 = session.xyDataObjects['RF:RF2 PI: CHD4-M-1 N: 409']
xy17 = session.xyDataObjects['RF:RF2 PI: CHD4-M-1 N: 410']
xy18 = session.xyDataObjects['RF:RF2 PI: CHD4-M-1 N: 411']
xy19 = session.xyDataObjects['RF:RF2 PI: CHD4-M-1 N: 412']
xy20 = session.xyDataObjects['RF:RF2 PI: CHD4-M-1 N: 413']
xy21 = session.xyDataObjects['RF:RF2 PI: CHD4-M-1 N: 3353']
xy22 = session.xyDataObjects['RF:RF2 PI: CHD4-M-1 N: 3354']
xy23 = session.xyDataObjects['RF:RF2 PI: CHD4-M-1 N: 3355']
xy24 = session.xyDataObjects['RF:RF2 PI: CHD4-M-1 N: 3356']
xy25 = session.xyDataObjects['RF:RF2 PI: CHD4-M-1 N: 3357']
xy26 = session.xyDataObjects['RF:RF2 PI: CHD4-M-1 N: 3358']
xy27 = session.xyDataObjects['RF:RF2 PI: CHD4-M-1 N: 3359']
xy28 = sum((xy1, xy2, xy3, xy4, xy5, xy6, xy7, xy8, xy9, xy10, xy11, xy12, 
    xy13, xy14, xy15, xy16, xy17, xy18, xy19, xy20, xy21, xy22, xy23, xy24, 
    xy25, xy26, xy27))*2
xy28.setValues(
    sourceDescription='sum ( ( "RF:RF2 PI: CHD4-M-1 N: 15", "RF:RF2 PI: CHD4-M-1 N: 16", "RF:RF2 PI: CHD4-M-1 N: 18", "RF:RF2 PI: CHD4-M-1 N: 19", "RF:RF2 PI: CHD4-M-1 N: 357", "RF:RF2 PI: CHD4-M-1 N: 381", "RF:RF2 PI: CHD4-M-1 N: 393", "RF:RF2 PI: CHD4-M-1 N: 394", "RF:RF2 PI: CHD4-M-1 N: 395", "RF:RF2 PI: CHD4-M-1 N: 396", "RF:RF2 PI: CHD4-M-1 N: 397", "RF:RF2 PI: CHD4-M-1 N: 398", "RF:RF2 PI: CHD4-M-1 N: 399", "RF:RF2 PI: CHD4-M-1 N: 407", "RF:RF2 PI: CHD4-M-1 N: 408", "RF:RF2 PI: CHD4-M-1 N: 409", "RF:RF2 PI: CHD4-M-1 N: 410", "RF:RF2 PI: CHD4-M-1 N: 411", "RF:RF2 PI: CHD4-M-1 N: 412", "RF:RF2 PI: CHD4-M-1 N: 413", "RF:RF2 PI: CHD4-M-1 N: 3353", "RF:RF2 PI: CHD4-M-1 N: 3354", "RF:RF2 PI: CHD4-M-1 N: 3355", "RF:RF2 PI: CHD4-M-1 N: 3356", "RF:RF2 PI: CHD4-M-1 N: 3357", "RF:RF2 PI: CHD4-M-1 N: 3358", "RF:RF2 PI: CHD4-M-1 N: 3359" ) ) * 2')
tmpName = xy28.name
session.xyDataObjects.changeKey(tmpName, 'force')
x0 = session.xyDataObjects['displacement']
x1 = session.xyDataObjects['force']
session.xyReportOptions.setValues(numDigits=9)
session.writeXYReport(fileName='FD_curve.txt', appendMode=OFF, xyData=(x0, x1))