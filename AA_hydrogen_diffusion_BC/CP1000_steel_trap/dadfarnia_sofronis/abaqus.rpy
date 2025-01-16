# -*- coding: mbcs -*-
#
# Abaqus/Viewer Release 2023.HF4 replay file
# Internal Version: 2023_07_21-20.45.57 RELr425 183702
# Run by nguyenb5 on Thu Sep  5 16:07:57 2024
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=90.2552032470703, 
    height=161.056716918945)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()
from viewerModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
o2 = session.openOdb(name='CP1000_diffusion_post.odb')
#: Model: C:/LocalUserData/User-data/nguyenb5/hydrogen_diffusion_BC/CP1000_steel_trap/dadfarnia_sofronis/CP1000_diffusion_post.odb
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
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0256061, 
    farPlane=0.0425644, width=0.0167222, height=0.00931442, 
    viewOffsetX=-0.00123232, viewOffsetY=-0.000609548)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='SDV_AR6_CT_DIS_MOL', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=242 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=242 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=242 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=242 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=242 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=242 )
session.viewports['Viewport: 1'].odbDisplay.setValues(viewCut=ON)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0257534, 
    farPlane=0.042417, width=0.0158093, height=0.00880593, 
    viewOffsetX=-0.00124799, viewOffsetY=-8.15871e-05)
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=242 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=242 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=242 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=242 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=242 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=242 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=242 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=242 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=242 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=0 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=1 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=2 )
session.viewports['Viewport: 1'].view.setValues(width=0.0149413, 
    height=0.00832248, viewOffsetX=-0.00103263, viewOffsetY=-7.39031e-05)
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=3 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=4 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=5 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=6 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=7 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=8 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=9 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=10 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=242 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=242 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=242 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=242 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=242 )
#: Warning: The output database 'C:/LocalUserData/User-data/nguyenb5/hydrogen_diffusion_BC/CP1000_steel_trap/dadfarnia_sofronis/CP1000_diffusion_post.odb' disk file has changed.
#: 
#: The current plot operation has been canceled, re-open the file to view the results
o1 = session.openOdb(
    name='C:/LocalUserData/User-data/nguyenb5/hydrogen_diffusion_BC/CP1000_steel_trap/dadfarnia_sofronis/CP1000_diffusion_post.odb')
session.viewports['Viewport: 1'].setValues(displayedObject=o1)
#: Model: C:/LocalUserData/User-data/nguyenb5/hydrogen_diffusion_BC/CP1000_steel_trap/dadfarnia_sofronis/CP1000_diffusion_post.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       5
#: Number of Node Sets:          5
#: Number of Steps:              1
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0257261, 
    farPlane=0.0424443, width=0.0178729, height=0.0099554, 
    viewOffsetX=0.000177021, viewOffsetY=-0.000687905)
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=282 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=283 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=284 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=285 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=286 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=294 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=299 )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='SDV_AR3_C_MOL', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=310 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=315 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=315 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=315 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=315 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=315 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=315 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=0 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=324 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=324 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=324 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=324 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=324 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=324 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=324 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=324 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=324 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=324 )
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0252949, 
    farPlane=0.0428755, width=0.0186951, height=0.0104133, 
    viewOffsetX=0.000762681, viewOffsetY=-0.000772373)
session.viewports['Viewport: 1'].odbDisplay.setValues(viewCut=ON)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['X-Plane'].setValues(
    position=0.0081)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['X-Plane'].setValues(
    position=0.008)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['X-Plane'].setValues(
    position=0.0075)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0270202, 
    farPlane=0.0411503, width=0.00795878, height=0.00443312, 
    viewOffsetX=-0.00167333, viewOffsetY=0.00100149)
session.viewports['Viewport: 1'].odbDisplay.setValues(viewCutNames=('Y-Plane', 
    ), viewCut=ON)
session.viewports['Viewport: 1'].odbDisplay.setValues(viewCutNames=('X-Plane', 
    ), viewCut=ON)
session.viewports['Viewport: 1'].odbDisplay.setValues(viewCutNames=('Y-Plane', 
    ), viewCut=ON)
session.viewports['Viewport: 1'].odbDisplay.setValues(viewCutNames=('X-Plane', 
    ), viewCut=ON)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0262318, 
    farPlane=0.0419386, width=0.0144039, height=0.00802314, 
    viewOffsetX=-0.00313356, viewOffsetY=0.000489542)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    showModelAboveCut=True)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    showFreeBodyCut=True)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    showFreeBodyCut=False)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    showModelAboveCut=False)
session.viewports['Viewport: 1'].odbDisplay.setValues(viewCutNames=('Z-Plane', 
    ), viewCut=ON)
session.viewports['Viewport: 1'].odbDisplay.setValues(viewCutNames=('Y-Plane', 
    ), viewCut=ON)
session.viewports['Viewport: 1'].odbDisplay.setValues(viewCutNames=('X-Plane', 
    ), viewCut=ON)
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=325 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=480 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=480 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=480 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=480 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=480 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=480 )
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    UNDEFORMED, ))
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=479 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=478 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=477 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=0 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=1 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=0 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=1 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=2 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=3 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=4 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=5 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=6 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=7 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=8 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=9 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=10 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=11 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=12 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=13 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=14 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=15 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=16 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=17 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=18 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=19 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=20 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=21 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=22 )
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0262312, 
    farPlane=0.0419393, width=0.0144036, height=0.00802294, 
    viewOffsetX=-0.00193874, viewOffsetY=0.000351429)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM1', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0264575, 
    farPlane=0.041713, width=0.0128368, height=0.00715023, 
    viewOffsetX=-0.00179598, viewOffsetY=0.000592312)
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=21 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=20 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=19 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=18 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=17 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=16 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=15 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=14 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=13 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=14 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=0 )
session.viewports['Viewport: 1'].view.setValues(width=0.0135996, 
    height=0.00757512, viewOffsetX=-0.00176799, viewOffsetY=0.000649508)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0262404, 
    farPlane=0.0419301, width=0.0135441, height=0.00754421, cameraPosition=(
    0.0278102, 0.0213973, 0.0198299), cameraUpVector=(-0.607197, 0.576558, 
    -0.546711), cameraTarget=(0.00813109, 0.00171815, 0.000150757), 
    viewOffsetX=-0.00176078, viewOffsetY=0.000646858)
session.viewports['Viewport: 1'].animationController.setValues(
    animationType=SCALE_FACTOR)
session.viewports['Viewport: 1'].animationController.play(duration=UNLIMITED)
session.viewports['Viewport: 1'].animationController.setValues(
    animationType=NONE)
session.viewports['Viewport: 1'].animationController.setValues(
    animationType=TIME_HISTORY)
session.viewports['Viewport: 1'].animationController.play(duration=UNLIMITED)
session.viewports['Viewport: 1'].animationController.setValues(
    animationType=NONE)
session.viewports['Viewport: 1'].animationController.setValues(
    animationType=TIME_HISTORY)
session.viewports['Viewport: 1'].animationController.play(duration=UNLIMITED)
session.viewports['Viewport: 1'].animationController.stop()
session.viewports['Viewport: 1'].animationController.showFirstFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.play(duration=UNLIMITED)
session.viewports['Viewport: 1'].animationController.stop()
session.viewports['Viewport: 1'].animationController.showFirstFrame()
session.viewports['Viewport: 1'].animationController.setValues(
    animationType=NONE)
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=1 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=2 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=3 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=4 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=5 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=6 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=7 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=480 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=479 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=478 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=477 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=476 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=475 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=474 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=473 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=472 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=471 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=0 )
session.viewports['Viewport: 1'].animationController.setValues(
    animationType=SCALE_FACTOR)
session.viewports['Viewport: 1'].animationController.play(duration=UNLIMITED)
session.viewports['Viewport: 1'].animationController.setValues(
    animationType=NONE)
session.viewports['Viewport: 1'].animationController.setValues(
    animationType=TIME_HISTORY)
session.viewports['Viewport: 1'].animationController.play(duration=UNLIMITED)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM2', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].animationController.stop()
session.viewports['Viewport: 1'].animationController.play(duration=UNLIMITED)
session.viewports['Viewport: 1'].animationController.stop()
session.viewports['Viewport: 1'].animationController.showFirstFrame()
session.viewports['Viewport: 1'].animationController.play(duration=UNLIMITED)
session.viewports['Viewport: 1'].animationController.stop()
session.viewports['Viewport: 1'].animationController.showFirstFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.play(duration=UNLIMITED)
session.viewports['Viewport: 1'].animationController.stop()
session.viewports['Viewport: 1'].animationController.showFirstFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.incrementFrame()
session.viewports['Viewport: 1'].animationController.setValues(
    animationType=NONE)
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=0 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=1 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=2 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=3 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=4 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=5 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=6 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=7 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=8 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=9 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=10 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=11 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=12 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=13 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=14 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=15 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=16 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=17 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=18 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=19 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=20 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=21 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=22 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=23 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=24 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=25 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=26 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=27 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=28 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=29 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=30 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=31 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=32 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=33 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=34 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=35 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=36 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=37 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=38 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=39 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=40 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=41 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=42 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=43 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=44 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=45 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=46 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=47 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=48 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=49 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=50 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=51 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=52 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=53 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=54 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=55 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=56 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=57 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=58 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=59 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=60 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=61 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=62 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=63 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=64 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=65 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=66 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=67 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=68 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=69 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=70 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=71 )
session.viewports['Viewport: 1'].animationController.setValues(
    animationType=TIME_HISTORY)
session.viewports['Viewport: 1'].animationController.play(duration=UNLIMITED)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.025728, 
    farPlane=0.0424425, width=0.0159883, height=0.00890564, 
    viewOffsetX=-0.00171127, viewOffsetY=0.000615276)
session.viewports['Viewport: 1'].animationController.stop()
session.viewports['Viewport: 1'].animationController.play(duration=UNLIMITED)
session.viewports['Viewport: 1'].animationController.stop()
session.viewports['Viewport: 1'].animationController.showFirstFrame()
session.viewports['Viewport: 1'].animationController.play(duration=UNLIMITED)
session.viewports['Viewport: 1'].animationController.stop()
session.viewports['Viewport: 1'].animationController.showFirstFrame()
session.viewports['Viewport: 1'].animationController.play(duration=UNLIMITED)
session.viewports['Viewport: 1'].animationController.stop()
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM1', outputPosition=INTEGRATION_POINT, )
session.animationOptions.setValues(timeHistoryMode=TIME_BASED, 
    minTimeAutoCompute=True, maxTimeAutoCompute=False, maxTime=1800)
session.animationOptions.setValues(frameRate=51)
session.viewports['Viewport: 1'].animationController.setValues(
    animationType=NONE)
session.viewports['Viewport: 1'].animationController.setValues(
    animationType=TIME_HISTORY)
session.viewports['Viewport: 1'].animationController.play(duration=UNLIMITED)
session.animationOptions.setValues(frameRate=100)
session.viewports['Viewport: 1'].animationController.stop()
session.viewports['Viewport: 1'].animationController.showFirstFrame()
session.viewports['Viewport: 1'].animationController.play(duration=UNLIMITED)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0264723, 
    farPlane=0.0416982, width=0.012844, height=0.00715423, 
    viewOffsetX=-0.00103455, viewOffsetY=0.000634005)
session.imageAnimationOptions.setValues(vpDecorations=ON, vpBackground=OFF, 
    compass=OFF)
session.writeImageAnimation(fileName='diffusion_TDS', format=AVI, 
    canvasObjects=(session.viewports['Viewport: 1'], ))
