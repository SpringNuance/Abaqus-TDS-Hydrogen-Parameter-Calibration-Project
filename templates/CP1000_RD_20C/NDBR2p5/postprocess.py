# -*- coding: mbcs -*-
#
# Abaqus/Viewer Release 2023.HF4 replay file
# Internal Version: 2023_07_21-20.45.57 RELr425 183702
# Run by nguyenb5 on Sun Jun  9 14:16:21 2024
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=119.81770324707, 
    height=100.520835876465)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()
from viewerModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
import os
os.chdir(os.getcwd())
o2 = session.openOdb(name='geometry.odb')
#: Model: C:/LocalUserData/User-data/nguyenb5/CP1000 processed/CP1000 NDBR2p5/Job-1.odb
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
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.456295, 
    farPlane=0.628462, width=0.0630981, height=0.0226859, 
    viewOffsetX=0.00532143, viewOffsetY=0.066355)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.459112, 
    farPlane=0.625645, width=0.0384527, height=0.013825, viewOffsetX=0.0012658, 
    viewOffsetY=0.0869083)
odb = session.odbs['geometry.odb']
session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('RF', 
    NODAL, ((COMPONENT, 'RF2'), )), ('U', NODAL, ((COMPONENT, 'U2'), )), ), 
    nodePick=(('NDBR2P5-M-1', 27, (
    '[#0:1223 #40000000 #600000 #a0000008 #80 #0:2 #4a00000', 
    ' #40c2a205 #92000701 #8 ]', )), ), )
xy1 = session.xyDataObjects['U:U2 PI: NDBR2P5-M-1 N: 39167']
xy2 = session.xyDataObjects['U:U2 PI: NDBR2P5-M-1 N: 39190']
xy3 = session.xyDataObjects['U:U2 PI: NDBR2P5-M-1 N: 39191']
xy4 = session.xyDataObjects['U:U2 PI: NDBR2P5-M-1 N: 39204']
xy5 = session.xyDataObjects['U:U2 PI: NDBR2P5-M-1 N: 39230']
xy6 = session.xyDataObjects['U:U2 PI: NDBR2P5-M-1 N: 39232']
xy7 = session.xyDataObjects['U:U2 PI: NDBR2P5-M-1 N: 39240']
xy8 = session.xyDataObjects['U:U2 PI: NDBR2P5-M-1 N: 39350']
xy9 = session.xyDataObjects['U:U2 PI: NDBR2P5-M-1 N: 39352']
xy10 = session.xyDataObjects['U:U2 PI: NDBR2P5-M-1 N: 39355']
xy11 = session.xyDataObjects['U:U2 PI: NDBR2P5-M-1 N: 39361']
xy12 = session.xyDataObjects['U:U2 PI: NDBR2P5-M-1 N: 39363']
xy13 = session.xyDataObjects['U:U2 PI: NDBR2P5-M-1 N: 39370']
xy14 = session.xyDataObjects['U:U2 PI: NDBR2P5-M-1 N: 39374']
xy15 = session.xyDataObjects['U:U2 PI: NDBR2P5-M-1 N: 39376']
xy16 = session.xyDataObjects['U:U2 PI: NDBR2P5-M-1 N: 39378']
xy17 = session.xyDataObjects['U:U2 PI: NDBR2P5-M-1 N: 39383']
xy18 = session.xyDataObjects['U:U2 PI: NDBR2P5-M-1 N: 39384']
xy19 = session.xyDataObjects['U:U2 PI: NDBR2P5-M-1 N: 39391']
xy20 = session.xyDataObjects['U:U2 PI: NDBR2P5-M-1 N: 39393']
xy21 = session.xyDataObjects['U:U2 PI: NDBR2P5-M-1 N: 39401']
xy22 = session.xyDataObjects['U:U2 PI: NDBR2P5-M-1 N: 39402']
xy23 = session.xyDataObjects['U:U2 PI: NDBR2P5-M-1 N: 39403']
xy24 = session.xyDataObjects['U:U2 PI: NDBR2P5-M-1 N: 39418']
xy25 = session.xyDataObjects['U:U2 PI: NDBR2P5-M-1 N: 39421']
xy26 = session.xyDataObjects['U:U2 PI: NDBR2P5-M-1 N: 39424']
xy27 = session.xyDataObjects['U:U2 PI: NDBR2P5-M-1 N: 39428']
xy28 = avg((xy1, xy2, xy3, xy4, xy5, xy6, xy7, xy8, xy9, xy10, xy11, xy12, 
    xy13, xy14, xy15, xy16, xy17, xy18, xy19, xy20, xy21, xy22, xy23, xy24, 
    xy25, xy26, xy27))
xy28.setValues(
    sourceDescription='avg ( ( "U:U2 PI: NDBR2P5-M-1 N: 39167", "U:U2 PI: NDBR2P5-M-1 N: 39190", "U:U2 PI: NDBR2P5-M-1 N: 39191", "U:U2 PI: NDBR2P5-M-1 N: 39204", "U:U2 PI: NDBR2P5-M-1 N: 39230", "U:U2 PI: NDBR2P5-M-1 N: 39232", "U:U2 PI: NDBR2P5-M-1 N: 39240", "U:U2 PI: NDBR2P5-M-1 N: 39350", "U:U2 PI: NDBR2P5-M-1 N: 39352", "U:U2 PI: NDBR2P5-M-1 N: 39355", "U:U2 PI: NDBR2P5-M-1 N: 39361", "U:U2 PI: NDBR2P5-M-1 N: 39363", "U:U2 PI: NDBR2P5-M-1 N: 39370", "U:U2 PI: NDBR2P5-M-1 N: 39374", "U:U2 PI: NDBR2P5-M-1 N: 39376", "U:U2 PI: NDBR2P5-M-1 N: 39378", "U:U2 PI: NDBR2P5-M-1 N: 39383", "U:U2 PI: NDBR2P5-M-1 N: 39384", "U:U2 PI: NDBR2P5-M-1 N: 39391", "U:U2 PI: NDBR2P5-M-1 N: 39393", "U:U2 PI: NDBR2P5-M-1 N: 39401", "U:U2 PI: NDBR2P5-M-1 N: 39402", "U:U2 PI: NDBR2P5-M-1 N: 39403", "U:U2 PI: NDBR2P5-M-1 N: 39418", "U:U2 PI: NDBR2P5-M-1 N: 39421", "U:U2 PI: NDBR2P5-M-1 N: 39424", "U:U2 PI: NDBR2P5-M-1 N: 39428" ) )')
tmpName = xy28.name
session.xyDataObjects.changeKey(tmpName, 'displacement')
xy1 = session.xyDataObjects['RF:RF2 PI: NDBR2P5-M-1 N: 39167']
xy2 = session.xyDataObjects['RF:RF2 PI: NDBR2P5-M-1 N: 39190']
xy3 = session.xyDataObjects['RF:RF2 PI: NDBR2P5-M-1 N: 39191']
xy4 = session.xyDataObjects['RF:RF2 PI: NDBR2P5-M-1 N: 39204']
xy5 = session.xyDataObjects['RF:RF2 PI: NDBR2P5-M-1 N: 39230']
xy6 = session.xyDataObjects['RF:RF2 PI: NDBR2P5-M-1 N: 39232']
xy7 = session.xyDataObjects['RF:RF2 PI: NDBR2P5-M-1 N: 39240']
xy8 = session.xyDataObjects['RF:RF2 PI: NDBR2P5-M-1 N: 39350']
xy9 = session.xyDataObjects['RF:RF2 PI: NDBR2P5-M-1 N: 39352']
xy10 = session.xyDataObjects['RF:RF2 PI: NDBR2P5-M-1 N: 39355']
xy11 = session.xyDataObjects['RF:RF2 PI: NDBR2P5-M-1 N: 39361']
xy12 = session.xyDataObjects['RF:RF2 PI: NDBR2P5-M-1 N: 39363']
xy13 = session.xyDataObjects['RF:RF2 PI: NDBR2P5-M-1 N: 39370']
xy14 = session.xyDataObjects['RF:RF2 PI: NDBR2P5-M-1 N: 39374']
xy15 = session.xyDataObjects['RF:RF2 PI: NDBR2P5-M-1 N: 39376']
xy16 = session.xyDataObjects['RF:RF2 PI: NDBR2P5-M-1 N: 39378']
xy17 = session.xyDataObjects['RF:RF2 PI: NDBR2P5-M-1 N: 39383']
xy18 = session.xyDataObjects['RF:RF2 PI: NDBR2P5-M-1 N: 39384']
xy19 = session.xyDataObjects['RF:RF2 PI: NDBR2P5-M-1 N: 39391']
xy20 = session.xyDataObjects['RF:RF2 PI: NDBR2P5-M-1 N: 39393']
xy21 = session.xyDataObjects['RF:RF2 PI: NDBR2P5-M-1 N: 39401']
xy22 = session.xyDataObjects['RF:RF2 PI: NDBR2P5-M-1 N: 39402']
xy23 = session.xyDataObjects['RF:RF2 PI: NDBR2P5-M-1 N: 39403']
xy24 = session.xyDataObjects['RF:RF2 PI: NDBR2P5-M-1 N: 39418']
xy25 = session.xyDataObjects['RF:RF2 PI: NDBR2P5-M-1 N: 39421']
xy26 = session.xyDataObjects['RF:RF2 PI: NDBR2P5-M-1 N: 39424']
xy27 = session.xyDataObjects['RF:RF2 PI: NDBR2P5-M-1 N: 39428']
xy28 = sum((xy1, xy2, xy3, xy4, xy5, xy6, xy7, xy8, xy9, xy10, xy11, xy12, 
    xy13, xy14, xy15, xy16, xy17, xy18, xy19, xy20, xy21, xy22, xy23, xy24, 
    xy25, xy26, xy27))*2
xy28.setValues(
    sourceDescription='sum ( ( "RF:RF2 PI: NDBR2P5-M-1 N: 39167", "RF:RF2 PI: NDBR2P5-M-1 N: 39190", "RF:RF2 PI: NDBR2P5-M-1 N: 39191", "RF:RF2 PI: NDBR2P5-M-1 N: 39204", "RF:RF2 PI: NDBR2P5-M-1 N: 39230", "RF:RF2 PI: NDBR2P5-M-1 N: 39232", "RF:RF2 PI: NDBR2P5-M-1 N: 39240", "RF:RF2 PI: NDBR2P5-M-1 N: 39350", "RF:RF2 PI: NDBR2P5-M-1 N: 39352", "RF:RF2 PI: NDBR2P5-M-1 N: 39355", "RF:RF2 PI: NDBR2P5-M-1 N: 39361", "RF:RF2 PI: NDBR2P5-M-1 N: 39363", "RF:RF2 PI: NDBR2P5-M-1 N: 39370", "RF:RF2 PI: NDBR2P5-M-1 N: 39374", "RF:RF2 PI: NDBR2P5-M-1 N: 39376", "RF:RF2 PI: NDBR2P5-M-1 N: 39378", "RF:RF2 PI: NDBR2P5-M-1 N: 39383", "RF:RF2 PI: NDBR2P5-M-1 N: 39384", "RF:RF2 PI: NDBR2P5-M-1 N: 39391", "RF:RF2 PI: NDBR2P5-M-1 N: 39393", "RF:RF2 PI: NDBR2P5-M-1 N: 39401", "RF:RF2 PI: NDBR2P5-M-1 N: 39402", "RF:RF2 PI: NDBR2P5-M-1 N: 39403", "RF:RF2 PI: NDBR2P5-M-1 N: 39418", "RF:RF2 PI: NDBR2P5-M-1 N: 39421", "RF:RF2 PI: NDBR2P5-M-1 N: 39424", "RF:RF2 PI: NDBR2P5-M-1 N: 39428" ) ) * 2')
tmpName = xy28.name
session.xyDataObjects.changeKey(tmpName, 'force')
xy1 = session.xyDataObjects['displacement']
xy2 = session.xyDataObjects['force']
xy3 = combine(xy1, xy2)
xyp = session.XYPlot('XYPlot-1')
chartName = xyp.charts.keys()[0]
chart = xyp.charts[chartName]
c1 = session.Curve(xyData=xy3)
chart.setValues(curvesToPlot=(c1, ), )
session.charts[chartName].autoColor(lines=True, symbols=True)
session.viewports['Viewport: 1'].setValues(displayedObject=xyp)
x0 = session.xyDataObjects['displacement']
x1 = session.xyDataObjects['force']
session.xyReportOptions.setValues(numDigits=9)
session.writeXYReport(fileName='FD_curve.txt', appendMode=OFF, xyData=(x0, x1))