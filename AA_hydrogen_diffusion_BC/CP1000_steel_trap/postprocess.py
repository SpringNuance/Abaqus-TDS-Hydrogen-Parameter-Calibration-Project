# -*- coding: mbcs -*-
#
# Abaqus/Viewer Release 2023.HF4 replay file
# Internal Version: 2023_07_21-20.45.57 RELr425 183702
# Run by nguyenb5 on Tue Sep  3 12:56:36 2024
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
import os
os.chdir(os.getcwd())
session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=90.2552032470703, 
    height=153.015045166016)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()
from viewerModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
o2 = session.openOdb(name='CP1000_diffusion_post.odb')
#: Model: C:/LocalUserData/User-data/nguyenb5/hydrogen_diffusion_BC/CP1000_steel_trap/CP1000_diffusion_post.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       5
#: Number of Node Sets:          5
#: Number of Steps:              1
session.viewports['Viewport: 1'].setValues(displayedObject=o2)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='SDV_AR2_DEQPLAS', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='SDV_AR3_C_MOL', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0248917, 
    farPlane=0.0432788, width=0.0236088, height=0.012388, 
    viewOffsetX=0.00093471, viewOffsetY=-0.000746784)
session.viewports[session.currentViewportName].odbDisplay.setFrame(
    step='Step-1', frame=8)
session.viewports[session.currentViewportName].odbDisplay.setFrame(
    step='Step-1', frame=40)
session.viewports[session.currentViewportName].odbDisplay.setFrame(
    step='Step-1', frame=120)
session.viewports[session.currentViewportName].odbDisplay.setFrame(
    step='Step-1', frame=480)
odbName=session.viewports[session.currentViewportName].odbDisplay.name
session.odbData[odbName].setValues(activeFrames=(('Step-1', (8, )), ))
odb = session.odbs['CP1000_steel_trap/CP1000_diffusion_post.odb']
session.xyDataListFromField(odb=odb, outputPosition=INTEGRATION_POINT, 
    variable=(('SDV_AR10_THETAL', INTEGRATION_POINT), ('SDV_AR11_THETAT_DIS', 
    INTEGRATION_POINT), ('SDV_AR12_THETAT_GB', INTEGRATION_POINT), (
    'SDV_AR13_THETAT_CARB', INTEGRATION_POINT), ('SDV_AR3_C_MOL', 
    INTEGRATION_POINT), ('SDV_AR4_CL_MOL', INTEGRATION_POINT), (
    'SDV_AR5_CT_MOL', INTEGRATION_POINT), ('SDV_AR6_CT_DIS_MOL', 
    INTEGRATION_POINT), ('SDV_AR7_CT_GB_MOL', INTEGRATION_POINT), (
    'SDV_AR8_CT_CARB_MOL', INTEGRATION_POINT), ('UVARM1', INTEGRATION_POINT), (
    'UVARM2', INTEGRATION_POINT), ('UVARM3', INTEGRATION_POINT), ('UVARM4', 
    INTEGRATION_POINT), ('UVARM5', INTEGRATION_POINT), ), operator=AVERAGE_ALL, 
    elementPick=(('PART-1-1', 4288, ('[#ffffffff:134 ]', )), ), )
#: Warning: Requested operation will result in the creation of a very large number of xyDataObjects. Performance can be affected. Please reduce the number of specified entities using Display Group operations before re-performing this operation.
x0 = session.xyDataObjects['AVERAGE_SDV_AR3_C_MOL']
x1 = session.xyDataObjects['AVERAGE_SDV_AR4_CL_MOL']
x2 = session.xyDataObjects['AVERAGE_SDV_AR5_CT_MOL']
x3 = session.xyDataObjects['AVERAGE_SDV_AR6_CT_DIS_MOL']
x4 = session.xyDataObjects['AVERAGE_SDV_AR7_CT_GB_MOL']
x5 = session.xyDataObjects['AVERAGE_SDV_AR8_CT_CARB_MOL']
x6 = session.xyDataObjects['AVERAGE_SDV_AR10_THETAL']
x7 = session.xyDataObjects['AVERAGE_SDV_AR11_THETAT_DIS']
x8 = session.xyDataObjects['AVERAGE_SDV_AR12_THETAT_GB']
x9 = session.xyDataObjects['AVERAGE_SDV_AR13_THETAT_CARB']
x10 = session.xyDataObjects['AVERAGE_UVARM1']
x11 = session.xyDataObjects['AVERAGE_UVARM2']
x12 = session.xyDataObjects['AVERAGE_UVARM3']
x13 = session.xyDataObjects['AVERAGE_UVARM4']
x14 = session.xyDataObjects['AVERAGE_UVARM5']
session.xyReportOptions.setValues(numDigits=9)
session.writeXYReport(fileName='results_2_mins.txt', xyData=(x0, x1, x2, x3, 
    x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14))
del session.xyDataObjects['AVERAGE_SDV_AR3_C_MOL']
del session.xyDataObjects['AVERAGE_SDV_AR4_CL_MOL']
del session.xyDataObjects['AVERAGE_SDV_AR5_CT_MOL']
del session.xyDataObjects['AVERAGE_SDV_AR6_CT_DIS_MOL']
del session.xyDataObjects['AVERAGE_SDV_AR7_CT_GB_MOL']
del session.xyDataObjects['AVERAGE_SDV_AR8_CT_CARB_MOL']
del session.xyDataObjects['AVERAGE_SDV_AR10_THETAL']
del session.xyDataObjects['AVERAGE_SDV_AR11_THETAT_DIS']
del session.xyDataObjects['AVERAGE_SDV_AR12_THETAT_GB']
del session.xyDataObjects['AVERAGE_SDV_AR13_THETAT_CARB']
del session.xyDataObjects['AVERAGE_UVARM1']
del session.xyDataObjects['AVERAGE_UVARM2']
del session.xyDataObjects['AVERAGE_UVARM3']
del session.xyDataObjects['AVERAGE_UVARM4']
del session.xyDataObjects['AVERAGE_UVARM5']
odb = session.odbs['C:/LocalUserData/User-data/nguyenb5/hydrogen_diffusion_BC/CP1000_steel_trap/CP1000_diffusion_post.odb']
session.xyDataListFromField(odb=odb, outputPosition=INTEGRATION_POINT, 
    variable=(('SDV_AR10_THETAL', INTEGRATION_POINT), ('SDV_AR11_THETAT_DIS', 
    INTEGRATION_POINT), ('SDV_AR12_THETAT_GB', INTEGRATION_POINT), (
    'SDV_AR13_THETAT_CARB', INTEGRATION_POINT), ('SDV_AR3_C_MOL', 
    INTEGRATION_POINT), ('SDV_AR4_CL_MOL', INTEGRATION_POINT), (
    'SDV_AR5_CT_MOL', INTEGRATION_POINT), ('SDV_AR6_CT_DIS_MOL', 
    INTEGRATION_POINT), ('SDV_AR7_CT_GB_MOL', INTEGRATION_POINT), (
    'SDV_AR8_CT_CARB_MOL', INTEGRATION_POINT), ('UVARM1', INTEGRATION_POINT), (
    'UVARM2', INTEGRATION_POINT), ('UVARM3', INTEGRATION_POINT), ('UVARM4', 
    INTEGRATION_POINT), ('UVARM5', INTEGRATION_POINT), ), operator=AVERAGE_ALL, 
    elementPick=(('PART-1-1', 4288, ('[#ffffffff:134 ]', )), ), )
#: Warning: Requested operation will result in the creation of a very large number of xyDataObjects. Performance can be affected. Please reduce the number of specified entities using Display Group operations before re-performing this operation.
x0 = session.xyDataObjects['AVERAGE_SDV_AR3_C_MOL']
x1 = session.xyDataObjects['AVERAGE_SDV_AR4_CL_MOL']
x2 = session.xyDataObjects['AVERAGE_SDV_AR5_CT_MOL']
x3 = session.xyDataObjects['AVERAGE_SDV_AR6_CT_DIS_MOL']
x4 = session.xyDataObjects['AVERAGE_SDV_AR7_CT_GB_MOL']
x5 = session.xyDataObjects['AVERAGE_SDV_AR8_CT_CARB_MOL']
x6 = session.xyDataObjects['AVERAGE_SDV_AR10_THETAL']
x7 = session.xyDataObjects['AVERAGE_SDV_AR11_THETAT_DIS']
x8 = session.xyDataObjects['AVERAGE_SDV_AR12_THETAT_GB']
x9 = session.xyDataObjects['AVERAGE_SDV_AR13_THETAT_CARB']
x10 = session.xyDataObjects['AVERAGE_UVARM1']
x11 = session.xyDataObjects['AVERAGE_UVARM2']
x12 = session.xyDataObjects['AVERAGE_UVARM3']
x13 = session.xyDataObjects['AVERAGE_UVARM4']
x14 = session.xyDataObjects['AVERAGE_UVARM5']
session.writeXYReport(fileName='results_10_mins.txt', xyData=(x0, x1, x2, x3, 
    x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14))
xyp = session.XYPlot('XYPlot-1')
chartName = xyp.charts.keys()[0]
chart = xyp.charts[chartName]
xy1 = session.xyDataObjects['AVERAGE_SDV_AR3_C_MOL']
c1 = session.Curve(xyData=xy1)
chart.setValues(curvesToPlot=(c1, ), )
session.charts[chartName].autoColor(lines=True, symbols=True)
session.viewports['Viewport: 1'].setValues(displayedObject=xyp)
xyp = session.xyPlots['XYPlot-1']
chartName = xyp.charts.keys()[0]
chart = xyp.charts[chartName]
xy1 = session.xyDataObjects['AVERAGE_SDV_AR4_CL_MOL']
c1 = session.Curve(xyData=xy1)
chart.setValues(curvesToPlot=(c1, ), )
session.charts[chartName].autoColor(lines=True, symbols=True)
xyp = session.xyPlots['XYPlot-1']
chartName = xyp.charts.keys()[0]
chart = xyp.charts[chartName]
xy1 = session.xyDataObjects['AVERAGE_SDV_AR4_CL_MOL']
c1 = session.Curve(xyData=xy1)
chart.setValues(curvesToPlot=(c1, ), )
session.charts[chartName].autoColor(lines=True, symbols=True)
x0 = session.xyDataObjects['AVERAGE_SDV_AR3_C_MOL']
x1 = session.xyDataObjects['AVERAGE_SDV_AR4_CL_MOL']
x2 = session.xyDataObjects['AVERAGE_SDV_AR5_CT_MOL']
x3 = session.xyDataObjects['AVERAGE_SDV_AR6_CT_DIS_MOL']
x4 = session.xyDataObjects['AVERAGE_SDV_AR7_CT_GB_MOL']
x5 = session.xyDataObjects['AVERAGE_SDV_AR8_CT_CARB_MOL']
x6 = session.xyDataObjects['AVERAGE_SDV_AR10_THETAL']
x7 = session.xyDataObjects['AVERAGE_SDV_AR11_THETAT_DIS']
x8 = session.xyDataObjects['AVERAGE_SDV_AR12_THETAT_GB']
x9 = session.xyDataObjects['AVERAGE_SDV_AR13_THETAT_CARB']
x10 = session.xyDataObjects['AVERAGE_UVARM1']
x11 = session.xyDataObjects['AVERAGE_UVARM2']
x12 = session.xyDataObjects['AVERAGE_UVARM3']
x13 = session.xyDataObjects['AVERAGE_UVARM4']
x14 = session.xyDataObjects['AVERAGE_UVARM5']
session.writeXYReport(fileName='results_10_mins.txt', xyData=(x0, x1, x2, x3, 
    x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14))
del session.xyDataObjects['AVERAGE_SDV_AR3_C_MOL']
del session.xyDataObjects['AVERAGE_SDV_AR4_CL_MOL']
del session.xyDataObjects['AVERAGE_SDV_AR5_CT_MOL']
del session.xyDataObjects['AVERAGE_SDV_AR6_CT_DIS_MOL']
del session.xyDataObjects['AVERAGE_SDV_AR7_CT_GB_MOL']
del session.xyDataObjects['AVERAGE_SDV_AR8_CT_CARB_MOL']
del session.xyDataObjects['AVERAGE_SDV_AR10_THETAL']
del session.xyDataObjects['AVERAGE_SDV_AR11_THETAT_DIS']
del session.xyDataObjects['AVERAGE_SDV_AR12_THETAT_GB']
del session.xyDataObjects['AVERAGE_SDV_AR13_THETAT_CARB']
del session.xyDataObjects['AVERAGE_UVARM1']
del session.xyDataObjects['AVERAGE_UVARM2']
del session.xyDataObjects['AVERAGE_UVARM3']
del session.xyDataObjects['AVERAGE_UVARM4']
del session.xyDataObjects['AVERAGE_UVARM5']
odb = session.odbs['C:/LocalUserData/User-data/nguyenb5/hydrogen_diffusion_BC/CP1000_steel_trap/CP1000_diffusion_post.odb']
session.viewports['Viewport: 1'].setValues(displayedObject=odb)
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
odbName=session.viewports[session.currentViewportName].odbDisplay.name
session.odbData[odbName].setValues(activeFrames=(('Step-1', (40, )), ))
odb = session.odbs['C:/LocalUserData/User-data/nguyenb5/hydrogen_diffusion_BC/CP1000_steel_trap/CP1000_diffusion_post.odb']
session.xyDataListFromField(odb=odb, outputPosition=INTEGRATION_POINT, 
    variable=(('SDV_AR10_THETAL', INTEGRATION_POINT), ('SDV_AR11_THETAT_DIS', 
    INTEGRATION_POINT), ('SDV_AR12_THETAT_GB', INTEGRATION_POINT), (
    'SDV_AR13_THETAT_CARB', INTEGRATION_POINT), ('SDV_AR3_C_MOL', 
    INTEGRATION_POINT), ('SDV_AR4_CL_MOL', INTEGRATION_POINT), (
    'SDV_AR5_CT_MOL', INTEGRATION_POINT), ('SDV_AR6_CT_DIS_MOL', 
    INTEGRATION_POINT), ('SDV_AR7_CT_GB_MOL', INTEGRATION_POINT), (
    'SDV_AR8_CT_CARB_MOL', INTEGRATION_POINT), ('UVARM1', INTEGRATION_POINT), (
    'UVARM2', INTEGRATION_POINT), ('UVARM3', INTEGRATION_POINT), ('UVARM4', 
    INTEGRATION_POINT), ('UVARM5', INTEGRATION_POINT), ), operator=AVERAGE_ALL, 
    elementPick=(('PART-1-1', 4288, ('[#ffffffff:134 ]', )), ), )
#: Warning: Requested operation will result in the creation of a very large number of xyDataObjects. Performance can be affected. Please reduce the number of specified entities using Display Group operations before re-performing this operation.
xyp = session.xyPlots['XYPlot-1']
chartName = xyp.charts.keys()[0]
chart = xyp.charts[chartName]
xy1 = session.xyDataObjects['AVERAGE_UVARM1']
c1 = session.Curve(xyData=xy1)
chart.setValues(curvesToPlot=(c1, ), )
session.charts[chartName].autoColor(lines=True, symbols=True)
session.viewports['Viewport: 1'].setValues(displayedObject=xyp)
x0 = session.xyDataObjects['AVERAGE_SDV_AR3_C_MOL']
x1 = session.xyDataObjects['AVERAGE_SDV_AR4_CL_MOL']
x2 = session.xyDataObjects['AVERAGE_SDV_AR5_CT_MOL']
x3 = session.xyDataObjects['AVERAGE_SDV_AR6_CT_DIS_MOL']
x4 = session.xyDataObjects['AVERAGE_SDV_AR7_CT_GB_MOL']
x5 = session.xyDataObjects['AVERAGE_SDV_AR8_CT_CARB_MOL']
x6 = session.xyDataObjects['AVERAGE_SDV_AR10_THETAL']
x7 = session.xyDataObjects['AVERAGE_SDV_AR11_THETAT_DIS']
x8 = session.xyDataObjects['AVERAGE_SDV_AR12_THETAT_GB']
x9 = session.xyDataObjects['AVERAGE_SDV_AR13_THETAT_CARB']
x10 = session.xyDataObjects['AVERAGE_UVARM1']
x11 = session.xyDataObjects['AVERAGE_UVARM2']
x12 = session.xyDataObjects['AVERAGE_UVARM3']
x13 = session.xyDataObjects['AVERAGE_UVARM4']
x14 = session.xyDataObjects['AVERAGE_UVARM5']
session.writeXYReport(fileName='results_10_mins.txt', appendMode=OFF, xyData=(
    x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14))
del session.xyDataObjects['AVERAGE_UVARM5']
del session.xyDataObjects['AVERAGE_SDV_AR3_C_MOL']
del session.xyDataObjects['AVERAGE_SDV_AR4_CL_MOL']
del session.xyDataObjects['AVERAGE_SDV_AR5_CT_MOL']
del session.xyDataObjects['AVERAGE_SDV_AR6_CT_DIS_MOL']
del session.xyDataObjects['AVERAGE_SDV_AR7_CT_GB_MOL']
del session.xyDataObjects['AVERAGE_SDV_AR8_CT_CARB_MOL']
del session.xyDataObjects['AVERAGE_SDV_AR10_THETAL']
del session.xyDataObjects['AVERAGE_SDV_AR11_THETAT_DIS']
del session.xyDataObjects['AVERAGE_SDV_AR12_THETAT_GB']
del session.xyDataObjects['AVERAGE_SDV_AR13_THETAT_CARB']
del session.xyDataObjects['AVERAGE_UVARM1']
del session.xyDataObjects['AVERAGE_UVARM2']
del session.xyDataObjects['AVERAGE_UVARM3']
del session.xyDataObjects['AVERAGE_UVARM4']
odb = session.odbs['C:/LocalUserData/User-data/nguyenb5/hydrogen_diffusion_BC/CP1000_steel_trap/CP1000_diffusion_post.odb']
session.viewports['Viewport: 1'].setValues(displayedObject=odb)
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
odbName=session.viewports[session.currentViewportName].odbDisplay.name
session.odbData[odbName].setValues(activeFrames=(('Step-1', (120, )), ))
odb = session.odbs['C:/LocalUserData/User-data/nguyenb5/hydrogen_diffusion_BC/CP1000_steel_trap/CP1000_diffusion_post.odb']
session.xyDataListFromField(odb=odb, outputPosition=INTEGRATION_POINT, 
    variable=(('SDV_AR10_THETAL', INTEGRATION_POINT), ('SDV_AR11_THETAT_DIS', 
    INTEGRATION_POINT), ('SDV_AR12_THETAT_GB', INTEGRATION_POINT), (
    'SDV_AR13_THETAT_CARB', INTEGRATION_POINT), ('SDV_AR3_C_MOL', 
    INTEGRATION_POINT), ('SDV_AR4_CL_MOL', INTEGRATION_POINT), (
    'SDV_AR5_CT_MOL', INTEGRATION_POINT), ('SDV_AR6_CT_DIS_MOL', 
    INTEGRATION_POINT), ('SDV_AR7_CT_GB_MOL', INTEGRATION_POINT), (
    'SDV_AR8_CT_CARB_MOL', INTEGRATION_POINT), ('UVARM1', INTEGRATION_POINT), (
    'UVARM2', INTEGRATION_POINT), ('UVARM3', INTEGRATION_POINT), ('UVARM4', 
    INTEGRATION_POINT), ('UVARM5', INTEGRATION_POINT), ), operator=AVERAGE_ALL, 
    elementPick=(('PART-1-1', 4288, ('[#ffffffff:134 ]', )), ), )
#: Warning: Requested operation will result in the creation of a very large number of xyDataObjects. Performance can be affected. Please reduce the number of specified entities using Display Group operations before re-performing this operation.
xyp = session.xyPlots['XYPlot-1']
chartName = xyp.charts.keys()[0]
chart = xyp.charts[chartName]
xy1 = session.xyDataObjects['AVERAGE_SDV_AR3_C_MOL']
c1 = session.Curve(xyData=xy1)
chart.setValues(curvesToPlot=(c1, ), )
session.charts[chartName].autoColor(lines=True, symbols=True)
session.viewports['Viewport: 1'].setValues(displayedObject=xyp)
xyp = session.xyPlots['XYPlot-1']
chartName = xyp.charts.keys()[0]
chart = xyp.charts[chartName]
xy1 = session.xyDataObjects['AVERAGE_UVARM1']
c1 = session.Curve(xyData=xy1)
chart.setValues(curvesToPlot=(c1, ), )
session.charts[chartName].autoColor(lines=True, symbols=True)
x0 = session.xyDataObjects['AVERAGE_SDV_AR3_C_MOL']
x1 = session.xyDataObjects['AVERAGE_SDV_AR4_CL_MOL']
x2 = session.xyDataObjects['AVERAGE_SDV_AR5_CT_MOL']
x3 = session.xyDataObjects['AVERAGE_SDV_AR6_CT_DIS_MOL']
x4 = session.xyDataObjects['AVERAGE_SDV_AR7_CT_GB_MOL']
x5 = session.xyDataObjects['AVERAGE_SDV_AR8_CT_CARB_MOL']
x6 = session.xyDataObjects['AVERAGE_SDV_AR10_THETAL']
x7 = session.xyDataObjects['AVERAGE_SDV_AR11_THETAT_DIS']
x8 = session.xyDataObjects['AVERAGE_SDV_AR12_THETAT_GB']
x9 = session.xyDataObjects['AVERAGE_SDV_AR13_THETAT_CARB']
x10 = session.xyDataObjects['AVERAGE_UVARM1']
x11 = session.xyDataObjects['AVERAGE_UVARM2']
x12 = session.xyDataObjects['AVERAGE_UVARM3']
x13 = session.xyDataObjects['AVERAGE_UVARM4']
x14 = session.xyDataObjects['AVERAGE_UVARM5']
session.writeXYReport(fileName='results_30_mins.txt', xyData=(x0, x1, x2, x3, 
    x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14))
del session.xyDataObjects['AVERAGE_UVARM5']
del session.xyDataObjects['AVERAGE_SDV_AR3_C_MOL']
del session.xyDataObjects['AVERAGE_SDV_AR4_CL_MOL']
del session.xyDataObjects['AVERAGE_SDV_AR5_CT_MOL']
del session.xyDataObjects['AVERAGE_SDV_AR6_CT_DIS_MOL']
del session.xyDataObjects['AVERAGE_SDV_AR7_CT_GB_MOL']
del session.xyDataObjects['AVERAGE_SDV_AR8_CT_CARB_MOL']
del session.xyDataObjects['AVERAGE_SDV_AR10_THETAL']
del session.xyDataObjects['AVERAGE_SDV_AR11_THETAT_DIS']
del session.xyDataObjects['AVERAGE_SDV_AR12_THETAT_GB']
del session.xyDataObjects['AVERAGE_SDV_AR13_THETAT_CARB']
del session.xyDataObjects['AVERAGE_UVARM1']
del session.xyDataObjects['AVERAGE_UVARM2']
del session.xyDataObjects['AVERAGE_UVARM3']
del session.xyDataObjects['AVERAGE_UVARM4']
odb = session.odbs['C:/LocalUserData/User-data/nguyenb5/hydrogen_diffusion_BC/CP1000_steel_trap/CP1000_diffusion_post.odb']
session.viewports['Viewport: 1'].setValues(displayedObject=odb)
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
odbName=session.viewports[session.currentViewportName].odbDisplay.name
session.odbData[odbName].setValues(activeFrames=(('Step-1', (480, )), ))
odb = session.odbs['C:/LocalUserData/User-data/nguyenb5/hydrogen_diffusion_BC/CP1000_steel_trap/CP1000_diffusion_post.odb']
session.xyDataListFromField(odb=odb, outputPosition=INTEGRATION_POINT, 
    variable=(('SDV_AR10_THETAL', INTEGRATION_POINT), ('SDV_AR11_THETAT_DIS', 
    INTEGRATION_POINT), ('SDV_AR12_THETAT_GB', INTEGRATION_POINT), (
    'SDV_AR13_THETAT_CARB', INTEGRATION_POINT), ('SDV_AR3_C_MOL', 
    INTEGRATION_POINT), ('SDV_AR4_CL_MOL', INTEGRATION_POINT), (
    'SDV_AR5_CT_MOL', INTEGRATION_POINT), ('SDV_AR6_CT_DIS_MOL', 
    INTEGRATION_POINT), ('SDV_AR7_CT_GB_MOL', INTEGRATION_POINT), (
    'SDV_AR8_CT_CARB_MOL', INTEGRATION_POINT), ('UVARM1', INTEGRATION_POINT), (
    'UVARM2', INTEGRATION_POINT), ('UVARM3', INTEGRATION_POINT), ('UVARM4', 
    INTEGRATION_POINT), ('UVARM5', INTEGRATION_POINT), ), operator=AVERAGE_ALL, 
    elementPick=(('PART-1-1', 4288, ('[#ffffffff:134 ]', )), ), )
#: Warning: Requested operation will result in the creation of a very large number of xyDataObjects. Performance can be affected. Please reduce the number of specified entities using Display Group operations before re-performing this operation.
xyp = session.xyPlots['XYPlot-1']
chartName = xyp.charts.keys()[0]
chart = xyp.charts[chartName]
xy1 = session.xyDataObjects['AVERAGE_UVARM1']
c1 = session.Curve(xyData=xy1)
chart.setValues(curvesToPlot=(c1, ), )
session.charts[chartName].autoColor(lines=True, symbols=True)
session.viewports['Viewport: 1'].setValues(displayedObject=xyp)
x0 = session.xyDataObjects['AVERAGE_SDV_AR3_C_MOL']
x1 = session.xyDataObjects['AVERAGE_SDV_AR4_CL_MOL']
x2 = session.xyDataObjects['AVERAGE_SDV_AR5_CT_MOL']
x3 = session.xyDataObjects['AVERAGE_SDV_AR6_CT_DIS_MOL']
x4 = session.xyDataObjects['AVERAGE_SDV_AR7_CT_GB_MOL']
x5 = session.xyDataObjects['AVERAGE_SDV_AR8_CT_CARB_MOL']
x6 = session.xyDataObjects['AVERAGE_SDV_AR10_THETAL']
x7 = session.xyDataObjects['AVERAGE_SDV_AR11_THETAT_DIS']
x8 = session.xyDataObjects['AVERAGE_SDV_AR12_THETAT_GB']
x9 = session.xyDataObjects['AVERAGE_SDV_AR13_THETAT_CARB']
x10 = session.xyDataObjects['AVERAGE_UVARM1']
x11 = session.xyDataObjects['AVERAGE_UVARM2']
x12 = session.xyDataObjects['AVERAGE_UVARM3']
x13 = session.xyDataObjects['AVERAGE_UVARM4']
x14 = session.xyDataObjects['AVERAGE_UVARM5']
session.writeXYReport(fileName='results_120_mins.txt', xyData=(x0, x1, x2, x3, 
    x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14))