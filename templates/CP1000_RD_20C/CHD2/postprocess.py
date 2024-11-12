# -*- coding: mbcs -*-
#
# Abaqus/Viewer Release 2023.HF4 replay file
# Internal Version: 2023_07_21-20.45.57 RELr425 183702
# Run by nguyenb5 on Sun Jun  9 17:02:51 2024
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
#: Model: C:/LocalUserData/User-data/nguyenb5/CP1000 processed/CP1000 CHD2/geometry.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       7
#: Number of Node Sets:          9
#: Number of Steps:              1
session.viewports['Viewport: 1'].setValues(displayedObject=o2)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.462679, 
    farPlane=0.622886, width=0.0213864, height=0.00967341, 
    viewOffsetX=-0.00149601, viewOffsetY=0.0865741)
odb = session.odbs['geometry.odb']
session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('RF', 
    NODAL, ((COMPONENT, 'RF2'), )), ('U', NODAL, ((COMPONENT, 'U2'), )), ), 
    nodePick=(('CHD2-M-1', 27, ('[#0:57 #36000 #0 #80fc0000 #3f #0 #20000000', 
    ' #200000 #2 #0:8 #1fc00 ]', )), ), )
xy1 = session.xyDataObjects['U:U2 PI: CHD2-M-1 N: 1838']
xy2 = session.xyDataObjects['U:U2 PI: CHD2-M-1 N: 1839']
xy3 = session.xyDataObjects['U:U2 PI: CHD2-M-1 N: 1841']
xy4 = session.xyDataObjects['U:U2 PI: CHD2-M-1 N: 1842']
xy5 = session.xyDataObjects['U:U2 PI: CHD2-M-1 N: 1907']
xy6 = session.xyDataObjects['U:U2 PI: CHD2-M-1 N: 1908']
xy7 = session.xyDataObjects['U:U2 PI: CHD2-M-1 N: 1909']
xy8 = session.xyDataObjects['U:U2 PI: CHD2-M-1 N: 1910']
xy9 = session.xyDataObjects['U:U2 PI: CHD2-M-1 N: 1911']
xy10 = session.xyDataObjects['U:U2 PI: CHD2-M-1 N: 1912']
xy11 = session.xyDataObjects['U:U2 PI: CHD2-M-1 N: 1920']
xy12 = session.xyDataObjects['U:U2 PI: CHD2-M-1 N: 1921']
xy13 = session.xyDataObjects['U:U2 PI: CHD2-M-1 N: 1922']
xy14 = session.xyDataObjects['U:U2 PI: CHD2-M-1 N: 1923']
xy15 = session.xyDataObjects['U:U2 PI: CHD2-M-1 N: 1924']
xy16 = session.xyDataObjects['U:U2 PI: CHD2-M-1 N: 1925']
xy17 = session.xyDataObjects['U:U2 PI: CHD2-M-1 N: 1926']
xy18 = session.xyDataObjects['U:U2 PI: CHD2-M-1 N: 2014']
xy19 = session.xyDataObjects['U:U2 PI: CHD2-M-1 N: 2038']
xy20 = session.xyDataObjects['U:U2 PI: CHD2-M-1 N: 2050']
xy21 = session.xyDataObjects['U:U2 PI: CHD2-M-1 N: 2347']
xy22 = session.xyDataObjects['U:U2 PI: CHD2-M-1 N: 2348']
xy23 = session.xyDataObjects['U:U2 PI: CHD2-M-1 N: 2349']
xy24 = session.xyDataObjects['U:U2 PI: CHD2-M-1 N: 2350']
xy25 = session.xyDataObjects['U:U2 PI: CHD2-M-1 N: 2351']
xy26 = session.xyDataObjects['U:U2 PI: CHD2-M-1 N: 2352']
xy27 = session.xyDataObjects['U:U2 PI: CHD2-M-1 N: 2353']
xy28 = avg((xy1, xy2, xy3, xy4, xy5, xy6, xy7, xy8, xy9, xy10, xy11, xy12, 
    xy13, xy14, xy15, xy16, xy17, xy18, xy19, xy20, xy21, xy22, xy23, xy24, 
    xy25, xy26, xy27))
xy28.setValues(
    sourceDescription='avg ( ( "U:U2 PI: CHD2-M-1 N: 1838", "U:U2 PI: CHD2-M-1 N: 1839", "U:U2 PI: CHD2-M-1 N: 1841", "U:U2 PI: CHD2-M-1 N: 1842", "U:U2 PI: CHD2-M-1 N: 1907", "U:U2 PI: CHD2-M-1 N: 1908", "U:U2 PI: CHD2-M-1 N: 1909", "U:U2 PI: CHD2-M-1 N: 1910", "U:U2 PI: CHD2-M-1 N: 1911", "U:U2 PI: CHD2-M-1 N: 1912", "U:U2 PI: CHD2-M-1 N: 1920", "U:U2 PI: CHD2-M-1 N: 1921", "U:U2 PI: CHD2-M-1 N: 1922", "U:U2 PI: CHD2-M-1 N: 1923", "U:U2 PI: CHD2-M-1 N: 1924", "U:U2 PI: CHD2-M-1 N: 1925", "U:U2 PI: CHD2-M-1 N: 1926", "U:U2 PI: CHD2-M-1 N: 2014", "U:U2 PI: CHD2-M-1 N: 2038", "U:U2 PI: CHD2-M-1 N: 2050", "U:U2 PI: CHD2-M-1 N: 2347", "U:U2 PI: CHD2-M-1 N: 2348", "U:U2 PI: CHD2-M-1 N: 2349", "U:U2 PI: CHD2-M-1 N: 2350", "U:U2 PI: CHD2-M-1 N: 2351", "U:U2 PI: CHD2-M-1 N: 2352", "U:U2 PI: CHD2-M-1 N: 2353" ) )')
tmpName = xy28.name
session.xyDataObjects.changeKey(tmpName, 'displacement')
xy1 = session.xyDataObjects['RF:RF2 PI: CHD2-M-1 N: 1838']
xy2 = session.xyDataObjects['RF:RF2 PI: CHD2-M-1 N: 1839']
xy3 = session.xyDataObjects['RF:RF2 PI: CHD2-M-1 N: 1841']
xy4 = session.xyDataObjects['RF:RF2 PI: CHD2-M-1 N: 1842']
xy5 = session.xyDataObjects['RF:RF2 PI: CHD2-M-1 N: 1907']
xy6 = session.xyDataObjects['RF:RF2 PI: CHD2-M-1 N: 1908']
xy7 = session.xyDataObjects['RF:RF2 PI: CHD2-M-1 N: 1909']
xy8 = session.xyDataObjects['RF:RF2 PI: CHD2-M-1 N: 1910']
xy9 = session.xyDataObjects['RF:RF2 PI: CHD2-M-1 N: 1911']
xy10 = session.xyDataObjects['RF:RF2 PI: CHD2-M-1 N: 1912']
xy11 = session.xyDataObjects['RF:RF2 PI: CHD2-M-1 N: 1920']
xy12 = session.xyDataObjects['RF:RF2 PI: CHD2-M-1 N: 1921']
xy13 = session.xyDataObjects['RF:RF2 PI: CHD2-M-1 N: 1922']
xy14 = session.xyDataObjects['RF:RF2 PI: CHD2-M-1 N: 1923']
xy15 = session.xyDataObjects['RF:RF2 PI: CHD2-M-1 N: 1924']
xy16 = session.xyDataObjects['RF:RF2 PI: CHD2-M-1 N: 1925']
xy17 = session.xyDataObjects['RF:RF2 PI: CHD2-M-1 N: 1926']
xy18 = session.xyDataObjects['RF:RF2 PI: CHD2-M-1 N: 2014']
xy19 = session.xyDataObjects['RF:RF2 PI: CHD2-M-1 N: 2038']
xy20 = session.xyDataObjects['RF:RF2 PI: CHD2-M-1 N: 2050']
xy21 = session.xyDataObjects['RF:RF2 PI: CHD2-M-1 N: 2347']
xy22 = session.xyDataObjects['RF:RF2 PI: CHD2-M-1 N: 2348']
xy23 = session.xyDataObjects['RF:RF2 PI: CHD2-M-1 N: 2349']
xy24 = session.xyDataObjects['RF:RF2 PI: CHD2-M-1 N: 2350']
xy25 = session.xyDataObjects['RF:RF2 PI: CHD2-M-1 N: 2351']
xy26 = session.xyDataObjects['RF:RF2 PI: CHD2-M-1 N: 2352']
xy27 = session.xyDataObjects['RF:RF2 PI: CHD2-M-1 N: 2353']
xy28 = sum((xy1, xy2, xy3, xy4, xy5, xy6, xy7, xy8, xy9, xy10, xy11, xy12, 
    xy13, xy14, xy15, xy16, xy17, xy18, xy19, xy20, xy21, xy22, xy23, xy24, 
    xy25, xy26, xy27))*2
xy28.setValues(
    sourceDescription='sum ( ( "RF:RF2 PI: CHD2-M-1 N: 1838", "RF:RF2 PI: CHD2-M-1 N: 1839", "RF:RF2 PI: CHD2-M-1 N: 1841", "RF:RF2 PI: CHD2-M-1 N: 1842", "RF:RF2 PI: CHD2-M-1 N: 1907", "RF:RF2 PI: CHD2-M-1 N: 1908", "RF:RF2 PI: CHD2-M-1 N: 1909", "RF:RF2 PI: CHD2-M-1 N: 1910", "RF:RF2 PI: CHD2-M-1 N: 1911", "RF:RF2 PI: CHD2-M-1 N: 1912", "RF:RF2 PI: CHD2-M-1 N: 1920", "RF:RF2 PI: CHD2-M-1 N: 1921", "RF:RF2 PI: CHD2-M-1 N: 1922", "RF:RF2 PI: CHD2-M-1 N: 1923", "RF:RF2 PI: CHD2-M-1 N: 1924", "RF:RF2 PI: CHD2-M-1 N: 1925", "RF:RF2 PI: CHD2-M-1 N: 1926", "RF:RF2 PI: CHD2-M-1 N: 2014", "RF:RF2 PI: CHD2-M-1 N: 2038", "RF:RF2 PI: CHD2-M-1 N: 2050", "RF:RF2 PI: CHD2-M-1 N: 2347", "RF:RF2 PI: CHD2-M-1 N: 2348", "RF:RF2 PI: CHD2-M-1 N: 2349", "RF:RF2 PI: CHD2-M-1 N: 2350", "RF:RF2 PI: CHD2-M-1 N: 2351", "RF:RF2 PI: CHD2-M-1 N: 2352", "RF:RF2 PI: CHD2-M-1 N: 2353" ) ) * 2')
tmpName = xy28.name
session.xyDataObjects.changeKey(tmpName, 'force')
x0 = session.xyDataObjects['displacement']
x1 = session.xyDataObjects['force']
session.xyReportOptions.setValues(numDigits=9)
session.writeXYReport(fileName='FD_curve.txt', appendMode=OFF, xyData=(x0, x1))