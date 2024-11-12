# -*- coding: mbcs -*-
#
# Abaqus/Viewer Release 2023.HF4 replay file
# Internal Version: 2023_07_21-20.45.57 RELr425 183702
# Run by nguyenb5 on Sun Jun  9 15:05:09 2024
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
#: Model: C:/LocalUserData/User-data/nguyenb5/CP1000 processed/CP1000 SH115/Job-1.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       21
#: Number of Node Sets:          23
#: Number of Steps:              1
session.viewports['Viewport: 1'].setValues(displayedObject=o2)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.46034, 
    farPlane=0.623989, width=0.0267694, height=0.0096245, 
    viewOffsetX=-0.00107158, viewOffsetY=0.0886346)
odb = session.odbs['geometry.odb']
session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('RF', 
    NODAL, ((COMPONENT, 'RF2'), )), ('U', NODAL, ((COMPONENT, 'U2'), )), ), 
    nodePick=(('SH115-M-1', 27, (
    '[#6c00 #0:3 #200000 #f0008000 #1fc07 #0:60 #fe ]', )), ), )
xy1 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 11']
xy2 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 12']
xy3 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 14']
xy4 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 15']
xy5 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 150']
xy6 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 176']
xy7 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 189']
xy8 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 190']
xy9 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 191']
xy10 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 192']
xy11 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 193']
xy12 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 194']
xy13 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 195']
xy14 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 203']
xy15 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 204']
xy16 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 205']
xy17 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 206']
xy18 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 207']
xy19 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 208']
xy20 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 209']
xy21 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 2146']
xy22 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 2147']
xy23 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 2148']
xy24 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 2149']
xy25 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 2150']
xy26 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 2151']
xy27 = session.xyDataObjects['U:U2 PI: SH115-M-1 N: 2152']
xy28 = avg((xy1, xy2, xy3, xy4, xy5, xy6, xy7, xy8, xy9, xy10, xy11, xy12, 
    xy13, xy14, xy15, xy16, xy17, xy18, xy19, xy20, xy21, xy22, xy23, xy24, 
    xy25, xy26, xy27))
xy28.setValues(
    sourceDescription='avg ( ( "U:U2 PI: SH115-M-1 N: 11", "U:U2 PI: SH115-M-1 N: 12", "U:U2 PI: SH115-M-1 N: 14", "U:U2 PI: SH115-M-1 N: 15", "U:U2 PI: SH115-M-1 N: 150", "U:U2 PI: SH115-M-1 N: 176", "U:U2 PI: SH115-M-1 N: 189", "U:U2 PI: SH115-M-1 N: 190", "U:U2 PI: SH115-M-1 N: 191", "U:U2 PI: SH115-M-1 N: 192", "U:U2 PI: SH115-M-1 N: 193", "U:U2 PI: SH115-M-1 N: 194", "U:U2 PI: SH115-M-1 N: 195", "U:U2 PI: SH115-M-1 N: 203", "U:U2 PI: SH115-M-1 N: 204", "U:U2 PI: SH115-M-1 N: 205", "U:U2 PI: SH115-M-1 N: 206", "U:U2 PI: SH115-M-1 N: 207", "U:U2 PI: SH115-M-1 N: 208", "U:U2 PI: SH115-M-1 N: 209", "U:U2 PI: SH115-M-1 N: 2146", "U:U2 PI: SH115-M-1 N: 2147", "U:U2 PI: SH115-M-1 N: 2148", "U:U2 PI: SH115-M-1 N: 2149", "U:U2 PI: SH115-M-1 N: 2150", "U:U2 PI: SH115-M-1 N: 2151", "U:U2 PI: SH115-M-1 N: 2152" ) )')
tmpName = xy28.name
session.xyDataObjects.changeKey(tmpName, 'displacement')
xy1 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 11']
xy2 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 12']
xy3 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 14']
xy4 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 15']
xy5 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 150']
xy6 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 176']
xy7 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 189']
xy8 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 190']
xy9 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 191']
xy10 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 192']
xy11 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 193']
xy12 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 194']
xy13 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 195']
xy14 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 203']
xy15 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 204']
xy16 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 205']
xy17 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 206']
xy18 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 207']
xy19 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 208']
xy20 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 209']
xy21 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 2146']
xy22 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 2147']
xy23 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 2148']
xy24 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 2149']
xy25 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 2150']
xy26 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 2151']
xy27 = session.xyDataObjects['RF:RF2 PI: SH115-M-1 N: 2152']
xy28 = sum((xy1, xy2, xy3, xy4, xy5, xy6, xy7, xy8, xy9, xy10, xy11, xy12, 
    xy13, xy14, xy15, xy16, xy17, xy18, xy19, xy20, xy21, xy22, xy23, xy24, 
    xy25, xy26, xy27))*2
xy28.setValues(
    sourceDescription='sum ( ( "RF:RF2 PI: SH115-M-1 N: 11", "RF:RF2 PI: SH115-M-1 N: 12", "RF:RF2 PI: SH115-M-1 N: 14", "RF:RF2 PI: SH115-M-1 N: 15", "RF:RF2 PI: SH115-M-1 N: 150", "RF:RF2 PI: SH115-M-1 N: 176", "RF:RF2 PI: SH115-M-1 N: 189", "RF:RF2 PI: SH115-M-1 N: 190", "RF:RF2 PI: SH115-M-1 N: 191", "RF:RF2 PI: SH115-M-1 N: 192", "RF:RF2 PI: SH115-M-1 N: 193", "RF:RF2 PI: SH115-M-1 N: 194", "RF:RF2 PI: SH115-M-1 N: 195", "RF:RF2 PI: SH115-M-1 N: 203", "RF:RF2 PI: SH115-M-1 N: 204", "RF:RF2 PI: SH115-M-1 N: 205", "RF:RF2 PI: SH115-M-1 N: 206", "RF:RF2 PI: SH115-M-1 N: 207", "RF:RF2 PI: SH115-M-1 N: 208", "RF:RF2 PI: SH115-M-1 N: 209", "RF:RF2 PI: SH115-M-1 N: 2146", "RF:RF2 PI: SH115-M-1 N: 2147", "RF:RF2 PI: SH115-M-1 N: 2148", "RF:RF2 PI: SH115-M-1 N: 2149", "RF:RF2 PI: SH115-M-1 N: 2150", "RF:RF2 PI: SH115-M-1 N: 2151", "RF:RF2 PI: SH115-M-1 N: 2152" ) ) * 2')
tmpName = xy28.name
session.xyDataObjects.changeKey(tmpName, 'force')
x0 = session.xyDataObjects['displacement']
x1 = session.xyDataObjects['force']
session.xyReportOptions.setValues(numDigits=9)
session.writeXYReport(fileName='FD_curve.txt', appendMode=OFF, xyData=(x0, x1))
x0 = session.xyDataObjects['displacement']
x1 = session.xyDataObjects['force']
session.writeXYReport(fileName='FD_curve.txt', xyData=(x0, x1))