# -*- coding: mbcs -*-
#
# Abaqus/Viewer Release 2023.HF4 replay file
# Internal Version: 2023_07_21-20.45.57 RELr425 183702
# Run by nguyenb5 on Wed Feb  5 13:51:03 2025
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=270.09375, 
    height=147.207168579102)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()
from viewerModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
o2 = session.openOdb(name='TDS_OneTrap.odb')
#: Model: C:/LocalUserData/User-data/nguyenb5/Abaqus-TDS-Hydrogen-Parameter-Calibration-Project/Abaqus-TDS/TDS_OneTrap.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       4
#: Number of Node Sets:          3
#: Number of Steps:              1
session.viewports['Viewport: 1'].setValues(displayedObject=o2)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].view.setValues(nearPlane=1.09661, 
    farPlane=1.73408, width=0.436603, height=0.227979, 
    viewOffsetX=-0.000579923, viewOffsetY=0.00480036)
session.viewports['Viewport: 1'].view.setValues(nearPlane=1.06753, 
    farPlane=1.76447, width=0.425024, height=0.221933, cameraPosition=(
    0.0463014, 0.134473, 1.39573), cameraUpVector=(-0.0747405, 0.992237, 
    -0.0993913), cameraTarget=(0.250453, 0.010114, 0.000718005), 
    viewOffsetX=-0.000564544, viewOffsetY=0.00467306)
session.viewports['Viewport: 1'].view.setValues(nearPlane=1.06947, 
    farPlane=1.76253, width=0.425796, height=0.222336, cameraPosition=(
    0.0456785, 0.134373, 1.39565), cameraUpVector=(0.0260499, 0.996042, 
    -0.0849804), cameraTarget=(0.24983, 0.0100145, 0.000635722), 
    viewOffsetX=-0.000565569, viewOffsetY=0.00468155)
session.viewports['Viewport: 1'].view.setValues(nearPlane=1.08375, 
    farPlane=1.74461, width=0.431481, height=0.225304, cameraPosition=(
    0.130755, -0.277497, 1.3795), cameraUpVector=(-0.162883, 0.968646, 0.1876), 
    cameraTarget=(0.251035, 0.0101114, -0.00108338), viewOffsetX=-0.00057312, 
    viewOffsetY=0.00474405)
session.viewports['Viewport: 1'].view.setValues(nearPlane=1.08337, 
    farPlane=1.74881, width=0.431331, height=0.225226, cameraPosition=(
    0.135567, 0.190002, 1.39994), cameraUpVector=(0.0440922, 0.991262, 
    -0.124318), cameraTarget=(0.249758, 0.00945464, 0.000810004), 
    viewOffsetX=-0.000572921, viewOffsetY=0.0047424)
session.viewports['Viewport: 1'].view.setValues(nearPlane=1.0834, 
    farPlane=1.74879, width=0.431342, height=0.225232, cameraPosition=(
    0.135872, 0.190036, 1.39996), cameraUpVector=(-0.00493576, 0.991713, 
    -0.128378), cameraTarget=(0.250063, 0.00948839, 0.000830512), 
    viewOffsetX=-0.000572936, viewOffsetY=0.00474252)
session.viewports['Viewport: 1'].view.setValues(nearPlane=1.08099, 
    farPlane=1.7489, width=0.430383, height=0.224731, cameraPosition=(0.133082, 
    -0.104541, 1.40544), cameraUpVector=(-0.0402187, 0.996194, 0.0773252), 
    cameraTarget=(0.250279, 0.00931823, -0.000436287), 
    viewOffsetX=-0.000571662, viewOffsetY=0.00473197)
session.viewports['Viewport: 1'].view.setValues(nearPlane=1.06373, 
    farPlane=1.76664, width=0.423513, height=0.221144, cameraPosition=(
    0.0199039, -0.0551008, 1.39484), cameraUpVector=(0.0460815, 0.997499, 
    0.0535892), cameraTarget=(0.249768, 0.00923072, -0.000230743), 
    viewOffsetX=-0.000562536, viewOffsetY=0.00465643)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='HFL', outputPosition=INTEGRATION_POINT, refinement=(
    INVARIANT, 'Magnitude'), )
session.viewports['Viewport: 1'].odbDisplay.display.setValues(
    plotState=CONTOURS_ON_DEF)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='NT11', outputPosition=NODAL, )
