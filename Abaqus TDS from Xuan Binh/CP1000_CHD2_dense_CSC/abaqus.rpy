# -*- coding: mbcs -*-
#
# Abaqus/Viewer Release 2023.HF4 replay file
# Internal Version: 2023_07_21-20.45.57 RELr425 183702
# Run by nguyenb5 on Fri Dec 20 16:17:13 2024
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Abaqus Warning: Unknown keyword (current_os) in environment file.
#: Abaqus Warning: Please check spelling of the environment variable names.
#:                 Unknown keyword "keywordname" can be removed using "del keywordname"
#:                 at the end of environment file.
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=113.546875, 
    height=150.557861328125)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()
from viewerModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
#: Abaqus Warning: Unknown keyword (current_os) in environment file.
#: Abaqus Warning: Please check spelling of the environment variable names.
#:                 Unknown keyword "keywordname" can be removed using "del keywordname"
#:                 at the end of environment file.
o2 = session.openOdb(name='CHD2_combined_processed.odb')
#: Model: C:/LocalUserData/User-data/nguyenb5/CP1000 plastic (UMAT UMATHT)/CP1000_dense_CSC/CP1000_CHD2_dense_CSC/CHD2_combined_processed.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       9
#: Number of Node Sets:          8
#: Number of Steps:              2
session.viewports['Viewport: 1'].setValues(displayedObject=o2)
session.viewports['Viewport: 1'].makeCurrent()
session.graphicsOptions.setValues(backgroundStyle=SOLID, 
    backgroundColor='#FFFFFF')
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.45594, 
    farPlane=0.629625, width=0.0659735, height=0.0349561, 
    viewOffsetX=0.000864423, viewOffsetY=-0.00948071)
session.viewports['Viewport: 1'].odbDisplay.commonOptions.setValues(
    visibleEdges=NONE)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.463479, 
    farPlane=0.622086, width=0.0171911, height=0.00910874, 
    viewOffsetX=0.000867468, viewOffsetY=-0.0101917)
session.viewports['Viewport: 1'].odbDisplay.setValues(viewCut=ON)
session.viewports['Viewport: 1'].odbDisplay.setValues(viewCutNames=('X-Plane', 
    ), viewCut=OFF)
session.viewports['Viewport: 1'].odbDisplay.setValues(viewCutNames=('Y-Plane', 
    ), viewCut=ON)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM1', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].odbDisplay.display.setValues(
    plotState=CONTOURS_ON_DEF)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.463943, 
    farPlane=0.620588, width=0.00749502, height=0.00397124, 
    viewOffsetX=-0.00167098, viewOffsetY=-0.0102288)
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=101 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=100 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=99 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=98 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=97 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=96 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=95 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=94 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=103 )
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.463118, 
    farPlane=0.621413, width=0.0128238, height=0.00679472, 
    viewOffsetX=-0.0024273, viewOffsetY=-0.00971349)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    position=0.0008955)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    position=0)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    motion=ROTATE)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    angle=349)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.460632, 
    farPlane=0.623899, width=0.032665, height=0.0173076, 
    viewOffsetX=-0.00690695, viewOffsetY=-0.00905966)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.462702, 
    farPlane=0.613528, width=0.0328118, height=0.0173853, cameraPosition=(
    0.453887, 0.28403, 0.059559), cameraUpVector=(-0.669953, 0.576126, 
    -0.468233), cameraTarget=(-0.00821184, 0.00705053, -0.00647346), 
    viewOffsetX=-0.00693799, viewOffsetY=-0.00910037)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.436812, 
    farPlane=0.649096, width=0.0309758, height=0.0164125, cameraPosition=(
    0.142177, 0.433174, 0.296557), cameraUpVector=(-0.153885, 0.29536, 
    -0.942912), cameraTarget=(0.00326589, 0.00502569, -0.00676904), 
    viewOffsetX=-0.00654978, viewOffsetY=-0.00859116)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.411602, 
    farPlane=0.672594, width=0.0291881, height=0.0154653, cameraPosition=(
    -0.00676808, 0.54302, 0.00248867), cameraUpVector=(-0.119798, -0.316934, 
    -0.940852), cameraTarget=(0.00279212, 0.000427303, -0.00825418), 
    viewOffsetX=-0.00617176, viewOffsetY=-0.00809533)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.414704, 
    farPlane=0.669491, width=0.00907604, height=0.00480894, 
    viewOffsetX=-0.00654331, viewOffsetY=-0.00625373)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.414646, 
    farPlane=0.668098, width=0.00907476, height=0.00480826, cameraPosition=(
    0.083606, 0.535212, 0.0261336), cameraUpVector=(-0.0433141, -0.271331, 
    -0.961511), cameraTarget=(0.00380281, -0.000634893, -0.00723036), 
    viewOffsetX=-0.00654239, viewOffsetY=-0.00625284)
session.viewports['Viewport: 1'].view.setValues(session.views['Top'])
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0653578, 
    farPlane=0.318454, width=0.00421713, height=0.00223445, 
    viewOffsetX=-0.000883235, viewOffsetY=0.000263924)
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=103 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=103 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=103 )
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0689038, 
    farPlane=0.313191, width=0.00444593, height=0.00235568, cameraPosition=(
    0.0617309, 0.181672, 0.0036747), cameraUpVector=(-0.60881, 0.229972, 
    -0.759252), cameraTarget=(-0.0204857, -0.0591815, -0.00335216), 
    viewOffsetX=-0.000931155, viewOffsetY=0.000278243)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.119458, 
    farPlane=0.25988, width=0.00770789, height=0.00408403, cameraPosition=(
    0.108852, 0.0999026, 0.119943), cameraUpVector=(-0.355514, 0.855722, 
    -0.375965), cameraTarget=(-0.0363658, -0.0317971, -0.0424959), 
    viewOffsetX=-0.00161434, viewOffsetY=0.000482389)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.16476, 
    farPlane=0.213871, width=0.0106309, height=0.00563281, cameraPosition=(
    0.0880528, 0.0314858, 0.165034), cameraUpVector=(-0.0918106, 0.987379, 
    -0.129045), cameraTarget=(-0.0290257, -0.00848267, -0.057485), 
    viewOffsetX=-0.00222655, viewOffsetY=0.000665325)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0999295, 
    farPlane=0.2799, width=0.0064478, height=0.00341639, cameraPosition=(
    0.0975623, 0.128352, 0.101764), cameraUpVector=(-0.205338, 0.70248, 
    -0.681438), cameraTarget=(-0.0323536, -0.0424802, -0.0351954), 
    viewOffsetX=-0.00135044, viewOffsetY=0.00040353)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.100196, 
    farPlane=0.279634, width=0.00828053, height=0.00438745, 
    viewOffsetX=-0.00172763, viewOffsetY=0.000141406)
session.viewports['Viewport: 1'].odbDisplay.setValues(viewCut=OFF)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.100782, 
    farPlane=0.279048, width=0.00448614, height=0.00237698, 
    viewOffsetX=-0.00156465, viewOffsetY=6.99553e-05)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.186495, 
    farPlane=0.195425, width=0.00830145, height=0.00439853, cameraPosition=(
    -0.0351611, -0.00219275, 0.187924), cameraUpVector=(0.180373, 0.982225, 
    0.0519672), cameraTarget=(0.013295, 0.00213064, -0.0619816), 
    viewOffsetX=-0.00289533, viewOffsetY=0.00012945)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.185939, 
    farPlane=0.19598, width=0.00827671, height=0.00438542, cameraPosition=(
    -0.0351119, -0.00192203, 0.187938), cameraUpVector=(0.0900955, 0.995329, 
    0.0346893), cameraTarget=(0.0133442, 0.00240136, -0.0619674), 
    viewOffsetX=-0.0028867, viewOffsetY=0.000129064)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0998528, 
    farPlane=0.284455, width=0.00444475, height=0.00235506, cameraPosition=(
    -0.0890963, 0.134874, 0.105311), cameraUpVector=(0.214407, 0.688286, 
    -0.693031), cameraTarget=(0.0308306, -0.0419264, -0.0331761), 
    viewOffsetX=-0.00155021, viewOffsetY=6.93099e-05)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.166862, 
    farPlane=0.216862, width=0.00742755, height=0.0039355, cameraPosition=(
    -0.0918026, 0.0315823, 0.165914), cameraUpVector=(0.21984, 0.973663, 
    -0.0604165), cameraTarget=(0.0316727, -0.00987104, -0.0528437), 
    viewOffsetX=-0.00259053, viewOffsetY=0.000115823)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.181649, 
    farPlane=0.204918, width=0.00808578, height=0.00428426, cameraPosition=(
    -0.157546, -0.0059557, -0.111557), cameraUpVector=(-0.0528341, 0.998512, 
    0.0134765), cameraTarget=(0.0475389, 0.00286304, 0.0390486), 
    viewOffsetX=-0.0028201, viewOffsetY=0.000126087)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.185196, 
    farPlane=0.201197, width=0.00824365, height=0.00436791, cameraPosition=(
    -0.041386, 0.00885375, -0.188319), cameraUpVector=(0.0650926, 0.997487, 
    0.0279847), cameraTarget=(0.00913708, -0.0014382, 0.0610028), 
    viewOffsetX=-0.00287516, viewOffsetY=0.000128549)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.166356, 
    farPlane=0.220129, width=0.00740503, height=0.00392356, cameraPosition=(
    -0.0938431, 0.0337025, -0.165488), cameraUpVector=(-0.165482, 0.945371, 
    0.280874), cameraTarget=(0.026292, -0.0106455, 0.0545586), 
    viewOffsetX=-0.00258267, viewOffsetY=0.000115472)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.139052, 
    farPlane=0.24757, width=0.00618963, height=0.00327958, cameraPosition=(
    -0.151975, 0.0726444, -0.0953237), cameraUpVector=(0.16113, 0.894984, 
    0.415983), cameraTarget=(0.0457061, -0.0228657, 0.0335938), 
    viewOffsetX=-0.00215877, viewOffsetY=9.65194e-05)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.180695, 
    farPlane=0.20603, width=0.00804326, height=0.00426173, cameraPosition=(
    -0.188374, 0.00763755, 0.0434795), cameraUpVector=(0.028733, 0.999072, 
    -0.032086), cameraTarget=(0.0605026, -0.0012202, -0.00945381), 
    viewOffsetX=-0.00280527, viewOffsetY=0.000125424)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.179743, 
    farPlane=0.207125, width=0.00800088, height=0.00423927, cameraPosition=(
    -0.0910149, 0.0140625, 0.170454), cameraUpVector=(0.0579861, 0.997233, 
    -0.0465182), cameraTarget=(0.0322903, -0.00346577, -0.0516016), 
    viewOffsetX=-0.00279049, viewOffsetY=0.000124763)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.177185, 
    farPlane=0.209683, width=0.027156, height=0.0143886, 
    viewOffsetX=-0.00239448, viewOffsetY=0.00106648)
session.odbs['C:/LocalUserData/User-data/nguyenb5/CP1000 plastic (UMAT UMATHT)/CP1000_dense_CSC/CP1000_CHD2_dense_CSC/CHD2_combined_processed.odb'].close(
    )
o1 = session.openOdb(
    name='C:/LocalUserData/User-data/nguyenb5/CP1000 plastic (UMAT UMATHT)/CP1000_dense_CSC/CP1000_CHD2_dense_CSC/CHD2_combined_processed.odb')
session.viewports['Viewport: 1'].setValues(displayedObject=o1)
#: Model: C:/LocalUserData/User-data/nguyenb5/CP1000 plastic (UMAT UMATHT)/CP1000_dense_CSC/CP1000_CHD2_dense_CSC/CHD2_combined_processed.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       9
#: Number of Node Sets:          8
#: Number of Steps:              2
o1 = session.openOdb(
    name='C:/LocalUserData/User-data/nguyenb5/CP1000 plastic (UMAT UMATHT)/CP1000_dense_CSC/CP1000_CHD4_dense_CSC/CHD4_combined_processed.odb')
session.viewports['Viewport: 1'].setValues(displayedObject=o1)
#: Model: C:/LocalUserData/User-data/nguyenb5/CP1000 plastic (UMAT UMATHT)/CP1000_dense_CSC/CP1000_CHD4_dense_CSC/CHD4_combined_processed.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       9
#: Number of Node Sets:          8
#: Number of Steps:              2
o1 = session.openOdb(
    name='C:/LocalUserData/User-data/nguyenb5/CP1000 plastic (UMAT UMATHT)/CP1000_dense_CSC/CP1000_NDBR15_dense_CSC/NDBR15_combined_processed.odb')
session.viewports['Viewport: 1'].setValues(displayedObject=o1)
#: Model: C:/LocalUserData/User-data/nguyenb5/CP1000 plastic (UMAT UMATHT)/CP1000_dense_CSC/CP1000_NDBR15_dense_CSC/NDBR15_combined_processed.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       9
#: Number of Node Sets:          8
#: Number of Steps:              2
o1 = session.openOdb(
    name='C:/LocalUserData/User-data/nguyenb5/CP1000 plastic (UMAT UMATHT)/CP1000_dense_CSC/CP1000_NDBR2p5_dense_CSC/NDBR2p5_combined_processed.odb')
session.viewports['Viewport: 1'].setValues(displayedObject=o1)
#: Model: C:/LocalUserData/User-data/nguyenb5/CP1000 plastic (UMAT UMATHT)/CP1000_dense_CSC/CP1000_NDBR2p5_dense_CSC/NDBR2p5_combined_processed.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       9
#: Number of Node Sets:          8
#: Number of Steps:              2
o1 = session.openOdb(
    name='C:/LocalUserData/User-data/nguyenb5/CP1000 plastic (UMAT UMATHT)/CP1000_dense_CSC/CP1000_NDBR40_dense_CSC/NDBR40_combined_processed.odb')
session.viewports['Viewport: 1'].setValues(displayedObject=o1)
#: Model: C:/LocalUserData/User-data/nguyenb5/CP1000 plastic (UMAT UMATHT)/CP1000_dense_CSC/CP1000_NDBR40_dense_CSC/NDBR40_combined_processed.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       9
#: Number of Node Sets:          8
#: Number of Steps:              2
o1 = session.openOdb(
    name='C:/LocalUserData/User-data/nguyenb5/CP1000 plastic (UMAT UMATHT)/CP1000_dense_CSC/CP1000_NDBR6_dense_CSC/NDBR6_combined_processed.odb')
session.viewports['Viewport: 1'].setValues(displayedObject=o1)
#: Model: C:/LocalUserData/User-data/nguyenb5/CP1000 plastic (UMAT UMATHT)/CP1000_dense_CSC/CP1000_NDBR6_dense_CSC/NDBR6_combined_processed.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       9
#: Number of Node Sets:          8
#: Number of Steps:              2
o1 = session.openOdb(
    name='C:/LocalUserData/User-data/nguyenb5/CP1000 plastic (UMAT UMATHT)/CP1000_dense_CSC/CP1000_SH115_dense_CSC/SH115_combined_processed.odb')
session.viewports['Viewport: 1'].setValues(displayedObject=o1)
#: Model: C:/LocalUserData/User-data/nguyenb5/CP1000 plastic (UMAT UMATHT)/CP1000_dense_CSC/CP1000_SH115_dense_CSC/SH115_combined_processed.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       9
#: Number of Node Sets:          8
#: Number of Steps:              2
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.44122, 
    farPlane=0.644345, width=0.16151, height=0.085576, viewOffsetX=-0.00484779, 
    viewOffsetY=-0.00307247)
o3 = session.odbs['C:/LocalUserData/User-data/nguyenb5/CP1000 plastic (UMAT UMATHT)/CP1000_dense_CSC/CP1000_CHD4_dense_CSC/CHD4_combined_processed.odb']
session.viewports['Viewport: 1'].setValues(displayedObject=o3)
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.463274, 
    farPlane=0.621419, width=0.0128261, height=0.00679591, 
    viewOffsetX=-0.000872659, viewOffsetY=-0.00944809)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM1', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.462965, 
    farPlane=0.621728, width=0.0167652, height=0.00888305, 
    viewOffsetX=-0.0033452, viewOffsetY=-0.00888692)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.46282, 
    farPlane=0.621873, width=0.01676, height=0.00888027, cameraPosition=(
    0.311528, 0.321806, 0.307043), cameraUpVector=(-0.328692, 0.531814, 
    -0.780472), cameraTarget=(-0.00184748, 0.00843053, -0.00633315), 
    viewOffsetX=-0.00334416, viewOffsetY=-0.00888414)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.46282, 
    farPlane=0.621872, width=0.01676, height=0.00888026, cameraPosition=(
    0.3131, 0.32064, 0.306637), cameraUpVector=(-0.1909, 0.473441, -0.85989), 
    cameraTarget=(-0.000275444, 0.00726471, -0.00673936), 
    viewOffsetX=-0.00334416, viewOffsetY=-0.00888413)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.506205, 
    farPlane=0.582177, width=0.0183311, height=0.0097127, cameraPosition=(
    0.0921079, 0.158004, 0.513126), cameraUpVector=(0.036262, 0.807571, 
    -0.588654), cameraTarget=(0.00154647, 0.0110141, -0.00146664), 
    viewOffsetX=-0.00365764, viewOffsetY=-0.00971693)
session.viewports['Viewport: 1'].odbDisplay.setValues(viewCut=ON)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    angle=8)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.43329, 
    farPlane=0.657098, width=0.0156906, height=0.00831369, cameraPosition=(
    -0.0530135, 0.475909, 0.262484), cameraUpVector=(-0.0528714, 0.175196, 
    -0.983113), cameraTarget=(-0.000743759, 0.00808057, -0.00773262), 
    viewOffsetX=-0.00313078, viewOffsetY=-0.00831728)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.417449, 
    farPlane=0.672772, width=0.015117, height=0.00800975, cameraPosition=(
    -0.0490175, 0.542209, 0.0413661), cameraUpVector=(-0.107869, -0.253316, 
    -0.961351), cameraTarget=(-0.00132559, 0.00400679, -0.0103712), 
    viewOffsetX=-0.00301632, viewOffsetY=-0.00801321)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.418996, 
    farPlane=0.670287, width=0.015173, height=0.00803944, cameraPosition=(
    0.0248245, 0.53588, 0.0991556), cameraUpVector=(-0.201114, -0.131007, 
    -0.970768), cameraTarget=(-0.0015618, 0.00486059, -0.0100926), 
    viewOffsetX=-0.0030275, viewOffsetY=-0.00804291)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.445604, 
    farPlane=0.643418, width=0.0161366, height=0.00854999, cameraPosition=(
    0.0478759, 0.422901, 0.341002), cameraUpVector=(-0.141445, 0.356014, 
    -0.923714), cameraTarget=(-0.000636064, 0.00912309, -0.00691635), 
    viewOffsetX=-0.00321976, viewOffsetY=-0.00855368)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.421467, 
    farPlane=0.668336, width=0.0152625, height=0.00808686, cameraPosition=(
    -0.0164123, 0.527149, 0.140514), cameraUpVector=(-0.109983, -0.0648863, 
    -0.991813), cameraTarget=(-0.000916701, 0.00574358, -0.0095147), 
    viewOffsetX=-0.00304536, viewOffsetY=-0.00809035)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.420975, 
    farPlane=0.668828, width=0.0183542, height=0.00972498, 
    viewOffsetX=-0.00325499, viewOffsetY=-0.00808753)
session.viewports['Viewport: 1'].odbDisplay.basicOptions.setValues(
    mirrorAboutXyPlane=True)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.421451, 
    farPlane=0.668489, width=0.0172725, height=0.00915184, 
    viewOffsetX=-0.0036171, viewOffsetY=-0.0083231)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.470577, 
    farPlane=0.618515, width=0.0192858, height=0.0102186, cameraPosition=(
    0.105732, 0.310627, 0.435244), cameraUpVector=(-0.111006, 0.598205, 
    -0.793617), cameraTarget=(0.000351244, 0.0102626, -0.00440184), 
    viewOffsetX=-0.00403872, viewOffsetY=-0.00929327)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.519075, 
    farPlane=0.570584, width=0.0212734, height=0.0112717, cameraPosition=(
    0.0956867, 0.102024, 0.526808), cameraUpVector=(0.0702134, 0.860216, 
    -0.505073), cameraTarget=(0.00227685, 0.0106574, -1.16562e-05), 
    viewOffsetX=-0.00445495, viewOffsetY=-0.010251)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.490521, 
    farPlane=0.602137, width=0.0201032, height=0.0106516, cameraPosition=(
    -0.119666, 0.230692, 0.481049), cameraUpVector=(-0.111556, 0.690555, 
    -0.714625), cameraTarget=(-0.00216075, 0.012121, -0.00168417), 
    viewOffsetX=-0.00420988, viewOffsetY=-0.00968709)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.439928, 
    farPlane=0.654814, width=0.0180297, height=0.00955297, cameraPosition=(
    -0.268553, 0.443308, 0.178248), cameraUpVector=(0.0708161, 0.0561675, 
    -0.995907), cameraTarget=(-0.00251289, 0.00883291, -0.00901189), 
    viewOffsetX=-0.00377566, viewOffsetY=-0.00868794)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.417239, 
    farPlane=0.673612, width=0.0170998, height=0.00906029, cameraPosition=(
    0.0669807, 0.540532, -0.0423891), cameraUpVector=(-0.170467, -0.366159, 
    -0.914805), cameraTarget=(0.000423318, 0.00267643, -0.0124848), 
    viewOffsetX=-0.00358094, viewOffsetY=-0.00823987)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.417293, 
    farPlane=0.674064, width=0.017102, height=0.00906147, cameraPosition=(
    0.013198, 0.546152, -0.0168606), cameraUpVector=(-0.118255, -0.338455, 
    -0.933522), cameraTarget=(0.000427113, 0.00353912, -0.0123287), 
    viewOffsetX=-0.00358141, viewOffsetY=-0.00824095)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.417942, 
    farPlane=0.673352, width=0.0171286, height=0.00907556, cameraPosition=(
    -0.00184295, 0.544959, 0.0415096), cameraUpVector=(-0.132998, -0.240099, 
    -0.961595), cameraTarget=(6.30873e-05, 0.00482864, -0.0120458), 
    viewOffsetX=-0.00358698, viewOffsetY=-0.00825376)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.417729, 
    farPlane=0.674076, width=0.0171199, height=0.00907094, cameraPosition=(
    -0.043063, 0.544484, 0.0259074), cameraUpVector=(-0.157655, -0.28054, 
    -0.946806), cameraTarget=(-0.000771846, 0.00470173, -0.0122914), 
    viewOffsetX=-0.00358515, viewOffsetY=-0.00824955)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.41709, 
    farPlane=0.674022, width=0.0170937, height=0.00905707, cameraPosition=(
    0.0307932, 0.545559, -0.00536687), cameraUpVector=(-0.191816, -0.310711, 
    -0.930948), cameraTarget=(-0.00020933, 0.00370965, -0.0124827), 
    viewOffsetX=-0.00357967, viewOffsetY=-0.00823693)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.42065, 
    farPlane=0.67005, width=0.0172396, height=0.00913437, cameraPosition=(
    0.0273159, 0.531076, 0.124658), cameraUpVector=(0.29211, -0.115415, 
    -0.949395), cameraTarget=(0.00504463, 0.00544593, -0.00886915), 
    viewOffsetX=-0.00361022, viewOffsetY=-0.00830723)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.455932, 
    farPlane=0.634788, width=0.0186856, height=0.0099005, cameraPosition=(
    -0.0801307, 0.375797, 0.387855), cameraUpVector=(0.694557, 0.341934, 
    -0.632986), cameraTarget=(0.00727931, 0.00740884, -0.00106904), 
    viewOffsetX=-0.00391302, viewOffsetY=-0.00900399)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.513824, 
    farPlane=0.574337, width=0.0210582, height=0.0111576, cameraPosition=(
    0.0414878, -0.105722, 0.532039), cameraUpVector=(0.317419, 0.934496, 
    -0.16113), cameraTarget=(0.00542993, 0.00887035, 0.00271848), 
    viewOffsetX=-0.00440987, viewOffsetY=-0.0101473)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.513478, 
    farPlane=0.574682, width=0.021044, height=0.0111501, cameraPosition=(
    0.0407928, -0.105111, 0.532219), cameraUpVector=(0.245805, 0.957441, 
    -0.151284), cameraTarget=(0.0047349, 0.00948098, 0.00289802), 
    viewOffsetX=-0.0044069, viewOffsetY=-0.0101405)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.461608, 
    farPlane=0.626627, width=0.0189182, height=0.0100237, cameraPosition=(
    0.0429881, 0.348535, 0.416391), cameraUpVector=(-0.0020313, 0.525899, 
    -0.850545), cameraTarget=(0.00214402, 0.00974814, -0.0057083), 
    viewOffsetX=-0.00396172, viewOffsetY=-0.00911613)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.477424, 
    farPlane=0.61101, width=0.0195664, height=0.0103671, cameraPosition=(
    0.0251248, 0.283717, 0.464316), cameraUpVector=(0.0224356, 0.645303, 
    -0.763597), cameraTarget=(0.0022598, 0.0105423, -0.00415509), 
    viewOffsetX=-0.00409746, viewOffsetY=-0.00942848)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.477649, 
    farPlane=0.610785, width=0.0184763, height=0.0097897, 
    viewOffsetX=-0.00517108, viewOffsetY=-0.00941535)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    angle=0)
session.viewports['Viewport: 1'].odbDisplay.ViewCut(name='Cut-4', shape=PLANE, 
    origin=(-0.002, 0., 0.), normal=(1.0, 0., 0.), axis2=(0.0, 1., 0.))
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.528633, 
    farPlane=0.551641, width=0.0204485, height=0.0108347, cameraPosition=(
    0.533783, -0.00310674, -0.0834868), cameraUpVector=(-0.424425, 0.563623, 
    -0.708656), cameraTarget=(-0.00425426, 0.00251123, -0.012094), 
    viewOffsetX=-0.00572305, viewOffsetY=-0.0104204)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.521067, 
    farPlane=0.558821, width=0.0201558, height=0.0106796, cameraPosition=(
    0.521871, 0.038808, 0.133801), cameraUpVector=(-0.144905, 0.511218, 
    -0.847147), cameraTarget=(0.000587176, 0.00190692, -0.0128767), 
    viewOffsetX=-0.00564114, viewOffsetY=-0.0102713)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.412841, 
    farPlane=0.66713, width=0.0159695, height=0.00846143, cameraPosition=(
    -0.0102022, 0.535294, 0.0768533), cameraUpVector=(-0.619977, -0.221075, 
    -0.752831), cameraTarget=(-0.00251371, -9.25275e-06, -0.0126147), 
    viewOffsetX=-0.00446947, viewOffsetY=-0.00813793)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.416508, 
    farPlane=0.663432, width=0.0161113, height=0.00853659, cameraPosition=(
    0.10061, 0.517641, 0.120138), cameraUpVector=(-0.596127, -0.0256916, 
    -0.802479), cameraTarget=(-0.0017331, 0.00142977, -0.0127726), 
    viewOffsetX=-0.00450917, viewOffsetY=-0.00821021)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.445937, 
    farPlane=0.634576, width=0.0172497, height=0.00913976, cameraPosition=(
    0.348459, 0.374774, -0.175271), cameraUpVector=(-0.878349, 0.151808, 
    -0.453273), cameraTarget=(-0.00751241, -0.000279429, -0.0102472), 
    viewOffsetX=-0.00482778, viewOffsetY=-0.00879032)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.520919, 
    farPlane=0.560037, width=0.0201502, height=0.0106766, cameraPosition=(
    0.352971, 0.058373, -0.405414), cameraUpVector=(-0.541476, 0.838018, 
    0.067306), cameraTarget=(-0.0092303, 0.00802788, -0.00430361), 
    viewOffsetX=-0.00563955, viewOffsetY=-0.0102684)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.507824, 
    farPlane=0.573685, width=0.0196437, height=0.0104082, cameraPosition=(
    -0.171385, -0.109273, -0.501084), cameraUpVector=(-0.306181, 0.920628, 
    0.242276), cameraTarget=(-0.00908448, 0.00849803, 0.00329866), 
    viewOffsetX=-0.00549778, viewOffsetY=-0.0100103)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.518861, 
    farPlane=0.562714, width=0.0200707, height=0.0106344, cameraPosition=(
    -0.325316, -0.0521909, -0.428912), cameraUpVector=(-0.0880641, 0.933156, 
    0.348517), cameraTarget=(-0.00653267, 0.00932736, 0.00606557), 
    viewOffsetX=-0.00561727, viewOffsetY=-0.0102279)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Cut-4'].setValues(
    normal=(0, 1, 0))
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.463543, 
    farPlane=0.617868, width=0.0179309, height=0.0095007, cameraPosition=(
    -0.125329, 0.319161, -0.418834), cameraUpVector=(0.107906, 0.574869, 
    0.8111), cameraTarget=(-0.00497501, 0.00761596, 0.00903), 
    viewOffsetX=-0.00501839, viewOffsetY=-0.00913747)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.508651, 
    farPlane=0.572725, width=0.0196758, height=0.0104252, cameraPosition=(
    -0.0257732, 0.134169, -0.523478), cameraUpVector=(-0.0134896, 0.84141, 
    0.540229), cameraTarget=(-0.0061771, 0.01043, 0.00464891), 
    viewOffsetX=-0.00550674, viewOffsetY=-0.0100267)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Cut-4'].setValues(
    origin=(0, -0.002, 0))
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.44211, 
    farPlane=0.639343, width=0.0171018, height=0.00906142, cameraPosition=(
    -0.239407, 0.399877, -0.275536), cameraUpVector=(-0.0524515, 0.215339, 
    0.97513), cameraTarget=(-0.00657402, 0.0016328, 0.010477), 
    viewOffsetX=-0.00478636, viewOffsetY=-0.00871502)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.524213, 
    farPlane=0.557277, width=0.0202777, height=0.0107442, cameraPosition=(
    -0.189706, -0.039513, -0.50492), cameraUpVector=(-0.272436, 0.888787, 
    0.368559), cameraTarget=(-0.0087292, 0.0082574, 0.00456865), 
    viewOffsetX=-0.00567522, viewOffsetY=-0.0103335)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.443425, 
    farPlane=0.637917, width=0.0171527, height=0.00908839, cameraPosition=(
    0.0964911, 0.406013, -0.344862), cameraUpVector=(-0.206903, 0.392128, 
    0.896341), cameraTarget=(-0.00777614, 0.0066537, 0.0076344), 
    viewOffsetX=-0.0048006, viewOffsetY=-0.00874098)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.477996, 
    farPlane=0.603372, width=0.01849, height=0.00979694, cameraPosition=(
    0.00581219, 0.26631, -0.471099), cameraUpVector=(-0.0227582, 0.672307, 
    0.739923), cameraTarget=(-0.00615186, 0.00912919, 0.00673753), 
    viewOffsetX=-0.00517487, viewOffsetY=-0.00942245)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Cut-4'].setValues(
    motion=ROTATE)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Cut-4'].setValues(
    angle=85)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Cut-4'].setValues(
    angle=360)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Cut-4'].setValues(
    angle=360)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Cut-4'].setValues(
    rotationAxis=AXIS_3)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Cut-4'].setValues(
    angle=315)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Cut-4'].setValues(
    angle=316)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Cut-4'].setValues(
    angle=360)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Cut-4'].setValues(
    motion=TRANSLATE)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Cut-4'].setValues(
    position=0.00275499)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.479039, 
    farPlane=0.602328, width=0.00938189, height=0.004971, 
    viewOffsetX=-0.00634048, viewOffsetY=-0.00933838)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Cut-4'].setValues(
    position=0)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.477584, 
    farPlane=0.603783, width=0.0187891, height=0.0099554, 
    viewOffsetX=-0.00458827, viewOffsetY=-0.0102254)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.488441, 
    farPlane=0.593211, width=0.0192162, height=0.0101817, cameraPosition=(
    -0.0347913, 0.220522, -0.493084), cameraUpVector=(0.0772151, 0.739018, 
    0.669245), cameraTarget=(-0.00506146, 0.00998696, 0.00631958), 
    viewOffsetX=-0.00469258, viewOffsetY=-0.0104578)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Cut-4'].setValues(
    origin=(-0.002, 0, 0))
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Cut-4'].setValues(
    motion=ROTATE)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Cut-4'].setValues(
    angle=0)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Cut-4'].setValues(
    angle=360)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Cut-4'].setValues(
    rotationAxis=AXIS_2)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Cut-4'].setValues(
    angle=360)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Cut-4'].setValues(
    angle=360)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Cut-4'].setValues(
    motion=TRANSLATE)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Cut-4'].setValues(
    position=-0.0118205)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Cut-4'].setValues(
    position=0)
#: 
#: Node: CHD4.127434
#:                                         1             2             3        Magnitude
#: Base coordinates:                  2.00512e-03, -3.63936e-04,  3.50000e-04,      -      
#: Scale:                             1.00000e+00,  1.00000e+00,  1.00000e+00,      -      
#: Deformed coordinates (unscaled):   1.96800e-03, -6.23707e-06,  3.17297e-04,      -      
#: Deformed coordinates (scaled):     1.96800e-03, -6.23707e-06,  3.17297e-04,      -      
#: Displacement (unscaled):          -3.71234e-05,  3.57698e-04, -3.27032e-05,  3.61104e-04
#: 
#: Node: CHD4.90013
#:                                         1             2             3        Magnitude
#: Base coordinates:                 -2.20354e-03, -3.72388e-04,  4.00000e-04,      -      
#: Scale:                             1.00000e+00,  1.00000e+00,  1.00000e+00,      -      
#: Deformed coordinates (unscaled):  -2.19291e-03, -5.92729e-06,  3.60034e-04,      -      
#: Deformed coordinates (scaled):    -2.19291e-03, -5.92729e-06,  3.60034e-04,      -      
#: Displacement (unscaled):           1.06280e-05,  3.66461e-04, -3.99656e-05,  3.68787e-04
#: 
#: Nodes for distance: CHD4.127434, CHD4.90013
#:                                        1             2             3        Magnitude
#: Base distance:                    -4.20866e-03, -8.45260e-06,  5.00000e-05,  4.20897e-03
#: Scale:                             1.00000e+00,  1.00000e+00,  1.00000e+00,      -      
#: Deformed distance (unscaled):     -4.16091e-03,  3.09781e-07,  4.27376e-05,  4.16113e-03
#: Deformed distance (scaled):       -4.16091e-03,  3.09781e-07,  4.27376e-05,  4.16113e-03
#: Relative displacement (unscaled):  4.77514e-05,  8.76238e-06, -7.26239e-06,  4.90889e-05
#: 
#: Node: CHD4.92419
#:                                         1             2             3        Magnitude
#: Base coordinates:                 -2.00632e-03, -3.54264e-04,  3.00000e-04,      -      
#: Scale:                             1.00000e+00,  1.00000e+00,  1.00000e+00,      -      
#: Deformed coordinates (unscaled):  -2.01256e-03, -4.66243e-08,  2.71847e-04,      -      
#: Deformed coordinates (scaled):    -2.01256e-03, -4.66243e-08,  2.71847e-04,      -      
#: Displacement (unscaled):          -6.23619e-06,  3.54218e-04, -2.81531e-05,  3.55389e-04
#: 
#: Node: CHD4.122130
#:                                         1             2             3        Magnitude
#: Base coordinates:                  1.98402e-03, -3.75402e-04,  5.00000e-04,      -      
#: Scale:                             1.00000e+00,  1.00000e+00,  1.00000e+00,      -      
#: Deformed coordinates (unscaled):   1.95149e-03, -4.87498e-06,  4.50507e-04,      -      
#: Deformed coordinates (scaled):     1.95149e-03, -4.87498e-06,  4.50507e-04,      -      
#: Displacement (unscaled):          -3.25253e-05,  3.70527e-04, -4.94933e-05,  3.75230e-04
#: 
#: Nodes for distance: CHD4.92419, CHD4.122130
#:                                        1             2             3        Magnitude
#: Base distance:                     3.99034e-03, -2.11378e-05,  2.00000e-04,  3.99541e-03
#: Scale:                             1.00000e+00,  1.00000e+00,  1.00000e+00,      -      
#: Deformed distance (unscaled):      3.96405e-03, -4.82835e-06,  1.78660e-04,  3.96808e-03
#: Deformed distance (scaled):        3.96405e-03, -4.82835e-06,  1.78660e-04,  3.96808e-03
#: Relative displacement (unscaled): -2.62891e-05,  1.63094e-05, -2.13402e-05,  3.75835e-05
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.489804, 
    farPlane=0.591849, width=0.00886293, height=0.00469603, 
    viewOffsetX=-0.00510745, viewOffsetY=-0.010315)
session.viewports['Viewport: 1'].odbDisplay.setValues(viewCutNames=('Cut-4', ), 
    viewCut=OFF)
#: 
#: Node: CHD4.105544
#:                                         1             2             3        Magnitude
#: Base coordinates:                  2.00000e-03,  0.00000e+00,  5.00000e-04,      -      
#: Scale:                             1.00000e+00,  1.00000e+00,  1.00000e+00,      -      
#: Deformed coordinates (unscaled):   2.20385e-03,  7.34897e-04,  2.23380e-04,      -      
#: Deformed coordinates (scaled):     2.20385e-03,  7.34897e-04,  2.23380e-04,      -      
#: Displacement (unscaled):           2.03848e-04,  7.34897e-04, -2.76620e-04,  8.11262e-04
#: 
#: Node: CHD4.90826
#:                                         1             2             3        Magnitude
#: Base coordinates:                 -2.11835e-03, -2.41889e-05,  3.50000e-04,      -      
#: Scale:                             1.00000e+00,  1.00000e+00,  1.00000e+00,      -      
#: Deformed coordinates (unscaled):  -2.25441e-03,  5.99801e-04,  1.69001e-04,      -      
#: Deformed coordinates (scaled):    -2.25441e-03,  5.99801e-04,  1.69001e-04,      -      
#: Displacement (unscaled):          -1.36066e-04,  6.23990e-04, -1.80999e-04,  6.63806e-04
#: 
#: Nodes for distance: CHD4.105544, CHD4.90826
#:                                        1             2             3        Magnitude
#: Base distance:                    -4.11835e-03, -2.41889e-05, -1.50000e-04,  4.12115e-03
#: Scale:                             1.00000e+00,  1.00000e+00,  1.00000e+00,      -      
#: Deformed distance (unscaled):     -4.45826e-03, -1.35095e-04, -5.43789e-05,  4.46064e-03
#: Deformed distance (scaled):       -4.45826e-03, -1.35095e-04, -5.43789e-05,  4.46064e-03
#: Relative displacement (unscaled): -3.39914e-04, -1.10906e-04,  9.56211e-05,  3.70115e-04
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.489963, 
    farPlane=0.591689, width=0.00783383, height=0.00415076, 
    viewOffsetX=-0.00572552, viewOffsetY=-0.00978865)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Cut-4'].setValues(
    origin=(0.002, 0, 0), axis2=(1, 0, 0))
session.viewports['Viewport: 1'].odbDisplay.setValues(viewCutNames=('Cut-4', ), 
    viewCut=ON)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.488828, 
    farPlane=0.592824, width=0.0171529, height=0.00684927, 
    viewOffsetX=-0.00609746, viewOffsetY=-0.0102742)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Cut-4'].setValues(
    motion=ROTATE)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Cut-4'].setValues(
    angle=286)
session.viewports['Viewport: 1'].odbDisplay.setValues(viewCut=OFF)
session.viewports['Viewport: 1'].odbDisplay.setValues(viewCutNames=('Y-Plane', 
    ), viewCut=ON)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    angle=157)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.488522, 
    farPlane=0.59313, width=0.0171421, height=0.00684498, 
    viewOffsetX=-0.00585124, viewOffsetY=-0.0105934)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.434792, 
    farPlane=0.648635, width=0.0152567, height=0.00609213, cameraPosition=(
    -0.192049, -0.429986, -0.266818), cameraUpVector=(-0.195776, 0.817395, 
    -0.54179), cameraTarget=(-0.00833793, 0.0087179, -0.0052807), 
    viewOffsetX=-0.00520768, viewOffsetY=-0.0094283)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.479678, 
    farPlane=0.603853, width=0.0168317, height=0.00672105, cameraPosition=(
    -0.0919736, -0.239696, -0.476861), cameraUpVector=(0.0325594, 0.988366, 
    -0.148567), cameraTarget=(-0.00561978, 0.011658, -0.00359903), 
    viewOffsetX=-0.00574529, viewOffsetY=-0.0104016)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.490558, 
    farPlane=0.593414, width=0.0172135, height=0.0068735, cameraPosition=(
    0.0615693, -0.193941, -0.50221), cameraUpVector=(-0.0281166, 0.99824, 
    -0.0522221), cameraTarget=(-0.00552287, 0.0114665, -0.00429546), 
    viewOffsetX=-0.00587561, viewOffsetY=-0.0106375)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.530376, 
    farPlane=0.55323, width=0.0186107, height=0.00743141, cameraPosition=(
    0.0229934, 0.0470812, -0.539482), cameraUpVector=(0.0130831, 0.918996, 
    0.394051), cameraTarget=(-0.00528768, 0.0124176, 0.00145376), 
    viewOffsetX=-0.00635252, viewOffsetY=-0.0115009)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.476608, 
    farPlane=0.605697, width=0.016724, height=0.00667803, cameraPosition=(
    -0.472286, -0.228485, -0.131932), cameraUpVector=(-0.16023, 0.971897, 
    0.17246), cameraTarget=(-0.0050664, 0.0099433, 0.00760656), 
    viewOffsetX=-0.00570852, viewOffsetY=-0.010335)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.475314, 
    farPlane=0.606992, width=0.0292268, height=0.0116705, 
    viewOffsetX=-0.00748892, viewOffsetY=-0.00904134)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.417228, 
    farPlane=0.664124, width=0.0256552, height=0.0102443, cameraPosition=(
    -0.0147804, -0.511822, -0.171862), cameraUpVector=(-0.904615, 0.403573, 
    -0.137116), cameraTarget=(-0.0122525, 0.00179502, 0.00364942), 
    viewOffsetX=-0.00657373, viewOffsetY=-0.00793644)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.518329, 
    farPlane=0.559104, width=0.0318719, height=0.0127267, cameraPosition=(
    -0.127651, 0.0602612, 0.520157), cameraUpVector=(-0.456631, -0.819682, 
    -0.345846), cameraTarget=(-0.0121561, -0.00261926, -0.00645433), 
    viewOffsetX=-0.00816666, viewOffsetY=-0.00985957)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.414635, 
    farPlane=0.661123, width=0.0254958, height=0.0101807, cameraPosition=(
    -0.11625, -0.501602, 0.153694), cameraUpVector=(0.236769, -0.0139422, 
    -0.971466), cameraTarget=(-0.00599533, 0.00307233, -0.0129216), 
    viewOffsetX=-0.00653288, viewOffsetY=-0.00788712)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.418597, 
    farPlane=0.656338, width=0.0257394, height=0.010278, cameraPosition=(
    0.209501, -0.479836, -0.119163), cameraUpVector=(-0.223995, 0.467029, 
    -0.8554), cameraTarget=(-0.0113233, 0.00358392, -0.00891632), 
    viewOffsetX=-0.0065953, viewOffsetY=-0.00796248)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.467256, 
    farPlane=0.608379, width=0.0287314, height=0.0114727, cameraPosition=(
    -0.311676, -0.273148, 0.342445), cameraUpVector=(0.401656, -0.647617, 
    -0.647507), cameraTarget=(-0.00187103, -0.004687, -0.0133112), 
    viewOffsetX=-0.00736196, viewOffsetY=-0.00888806)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.482327, 
    farPlane=0.592934, width=0.0296581, height=0.0118427, cameraPosition=(
    -0.116605, -0.219186, 0.476719), cameraUpVector=(0.186491, -0.736848, 
    -0.649828), cameraTarget=(-0.00623545, -0.00706788, -0.010557), 
    viewOffsetX=-0.00759941, viewOffsetY=-0.00917473)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.502789, 
    farPlane=0.571892, width=0.0309163, height=0.0123451, cameraPosition=(
    0.327419, -0.117925, 0.409427), cameraUpVector=(-0.241184, -0.850586, 
    -0.467263), cameraTarget=(-0.0105404, -0.00925223, -0.00116643), 
    viewOffsetX=-0.00792181, viewOffsetY=-0.00956396)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.515118, 
    farPlane=0.559297, width=0.0316744, height=0.0126478, cameraPosition=(
    0.48803, 0.0368109, 0.222038), cameraUpVector=(-0.292881, -0.955448, 
    0.0366053), cameraTarget=(-0.00843723, -0.00859321, 0.00739445), 
    viewOffsetX=-0.00811606, viewOffsetY=-0.00979848)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.517243, 
    farPlane=0.557172, width=0.0173518, height=0.0069287, 
    viewOffsetX=-0.00937899, viewOffsetY=-0.0113626)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.4478, 
    farPlane=0.623881, width=0.0150222, height=0.00599848, cameraPosition=(
    0.227628, -0.358386, 0.326402), cameraUpVector=(-0.299362, -0.475127, 
    -0.827428), cameraTarget=(-0.012514, -0.00496745, -0.00832151), 
    viewOffsetX=-0.00811981, viewOffsetY=-0.00983713)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.450223, 
    farPlane=0.619695, width=0.0151035, height=0.00603094, cameraPosition=(
    -0.225029, -0.339671, 0.346223), cameraUpVector=(-0.0750841, -0.391633, 
    -0.917053), cameraTarget=(-0.00743509, 0.00218942, -0.0148814), 
    viewOffsetX=-0.00816375, viewOffsetY=-0.00989037)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.433926, 
    farPlane=0.635949, width=0.0145568, height=0.00581264, cameraPosition=(
    -0.249147, -0.408417, 0.238507), cameraUpVector=(0.293652, -0.299616, 
    -0.907744), cameraTarget=(-0.00253598, 0.00232877, -0.0166073), 
    viewOffsetX=-0.00786825, viewOffsetY=-0.00953237)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.419944, 
    farPlane=0.649894, width=0.0140878, height=0.00562535, cameraPosition=(
    -0.264861, -0.458341, 0.0736827), cameraUpVector=(-0.273765, 0.381388, 
    -0.882947), cameraTarget=(-0.00833103, 0.0131191, -0.00712559), 
    viewOffsetX=-0.00761473, viewOffsetY=-0.00922523)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.420013, 
    farPlane=0.649826, width=0.0140901, height=0.00562627, cameraPosition=(
    -0.2624, -0.460465, 0.0691021), cameraUpVector=(0.0128464, 0.209181, 
    -0.977793), cameraTarget=(-0.00587006, 0.0109949, -0.0117062), 
    viewOffsetX=-0.00761598, viewOffsetY=-0.00922674)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.408109, 
    farPlane=0.662139, width=0.0136908, height=0.00546681, cameraPosition=(
    -0.107971, -0.523548, 0.0045817), cameraUpVector=(-0.0260331, 0.316493, 
    -0.948237), cameraTarget=(-0.00850331, 0.00981603, -0.0109343), 
    viewOffsetX=-0.00740013, viewOffsetY=-0.00896523)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.408261, 
    farPlane=0.661986, width=0.0138727, height=0.00553947, 
    viewOffsetX=-0.00489948, viewOffsetY=-0.00875257)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.406803, 
    farPlane=0.662562, width=0.0138232, height=0.00551969, cameraPosition=(
    -0.0189021, -0.533667, -0.0117276), cameraUpVector=(-0.105972, 0.337241, 
    -0.935435), cameraTarget=(-0.0101691, 0.00904266, -0.0104779), 
    viewOffsetX=-0.00488199, viewOffsetY=-0.00872132)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.406811, 
    farPlane=0.662555, width=0.0138235, height=0.00551979, cameraPosition=(
    -0.0170688, -0.533694, -0.012844), cameraUpVector=(0.0479648, 0.334777, 
    -0.941076), cameraTarget=(-0.00833585, 0.00901573, -0.0115943), 
    viewOffsetX=-0.00488208, viewOffsetY=-0.00872148)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.406381, 
    farPlane=0.662984, width=0.0178317, height=0.00712035, 
    viewOffsetX=-0.00555171, viewOffsetY=-0.00862634)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.406074, 
    farPlane=0.663129, width=0.0178183, height=0.00711497, cameraPosition=(
    0.00876257, -0.53377, -0.0145225), cameraUpVector=(0.0358549, 0.339732, 
    -0.939839), cameraTarget=(-0.00867436, 0.00872316, -0.0115992), 
    viewOffsetX=-0.00554751, viewOffsetY=-0.00861982)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.427218, 
    farPlane=0.642263, width=0.0187461, height=0.00748544, cameraPosition=(
    -0.0287891, -0.449322, 0.287673), cameraUpVector=(0.0758974, -0.249385, 
    -0.965426), cameraTarget=(-0.00788894, 0.00111857, -0.0144495), 
    viewOffsetX=-0.00583637, viewOffsetY=-0.00906865)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.426087, 
    farPlane=0.643394, width=0.0272123, height=0.0108661, 
    viewOffsetX=-0.00584587, viewOffsetY=-0.00939327)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.463052, 
    farPlane=0.607591, width=0.0295731, height=0.0118088, cameraPosition=(
    -0.153989, -0.289506, 0.422847), cameraUpVector=(0.358333, -0.623293, 
    -0.695057), cameraTarget=(-0.00285017, -0.0054439, -0.0142772), 
    viewOffsetX=-0.00635302, viewOffsetY=-0.0102082)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.51427, 
    farPlane=0.555854, width=0.0328442, height=0.013115, cameraPosition=(
    0.0180707, -0.0785258, 0.529029), cameraUpVector=(0.23353, -0.859518, 
    -0.454635), cameraTarget=(-0.00580651, -0.0109306, -0.008998), 
    viewOffsetX=-0.00705573, viewOffsetY=-0.0113373)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.513795, 
    farPlane=0.556331, width=0.0328138, height=0.0131029, cameraPosition=(
    0.0144561, -0.0768816, 0.529396), cameraUpVector=(-0.0347997, -0.893838, 
    -0.447038), cameraTarget=(-0.00942116, -0.00928638, -0.00863101), 
    viewOffsetX=-0.00704921, viewOffsetY=-0.0113268)
session.viewports['Viewport: 1'].view.setValues(session.views['Back'])
session.viewports['Viewport: 1'].view.setValues(session.views['Front'])
session.viewports['Viewport: 1'].view.setValues(session.views['Back'])
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.257315, 
    farPlane=0.264778, width=0.0208821, height=0.00833836, 
    viewOffsetX=-0.00125955, viewOffsetY=-0.0609963)
session.viewports['Viewport: 1'].view.setValues(session.views['Back'])
session.viewports['Viewport: 1'].view.setValues(session.views['Front'])
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.11189, 
    farPlane=0.345552, width=0.14641, height=0.0584628, cameraPosition=(
    -0.0101502, -0.0768142, 0.221045), cameraUpVector=(0.073703, 0.843572, 
    0.531935), cameraTarget=(7.13766e-06, 0.0616834, 0))
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0773307, 
    farPlane=0.282149, width=0.101189, height=0.0404055, cameraPosition=(
    -0.00918525, -0.101801, -0.148231), cameraUpVector=(-0.0901606, -0.763091, 
    0.639972), cameraTarget=(-0.000129245, 0.0652149, 0.0521915))
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.173434, 
    farPlane=0.206172, width=0.226943, height=0.0906198, cameraPosition=(
    0.00143072, 0.0118247, -0.189914), cameraUpVector=(-0.0609771, -0.99812, 
    0.00612625), cameraTarget=(-0.00493149, 0.0138152, 0.0710474))
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.143378, 
    farPlane=0.234755, width=0.187613, height=0.0749152, cameraPosition=(
    -0.017741, -0.00421919, -0.188608), cameraUpVector=(-0.0569125, -0.993692, 
    0.0966307), cameraTarget=(0.00226471, 0.0198374, 0.0705571))
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.171316, 
    farPlane=0.206817, width=0.0329262, height=0.0131477, 
    viewOffsetX=0.00201261, viewOffsetY=0.00797102)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.12506, 
    farPlane=0.250361, width=0.0240361, height=0.00959778, cameraPosition=(
    -0.0139108, -0.0738958, -0.172135), cameraUpVector=(0.137093, -0.885415, 
    0.444123), cameraTarget=(-0.00153149, 0.0445437, 0.0601671), 
    viewOffsetX=0.0014692, viewOffsetY=0.00581882)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.141932, 
    farPlane=0.234723, width=0.0272788, height=0.0108926, cameraPosition=(
    -0.0345078, -0.0497627, -0.178563), cameraUpVector=(0.136647, -0.941385, 
    0.308418), cameraTarget=(0.0065558, 0.035863, 0.0645987), 
    viewOffsetX=0.00166741, viewOffsetY=0.00660383)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.14146, 
    farPlane=0.235195, width=0.0271881, height=0.0108564, cameraPosition=(
    -0.0330145, -0.0494029, -0.178942), cameraUpVector=(-0.0360034, -0.940697, 
    0.337332), cameraTarget=(0.00804909, 0.0362228, 0.0642198), 
    viewOffsetX=0.00166187, viewOffsetY=0.00658187)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.180724, 
    farPlane=0.19823, width=0.0347346, height=0.0138698, cameraPosition=(
    -0.0294137, 0.0176971, -0.186824), cameraUpVector=(-0.0758442, -0.997002, 
    -0.015284), cameraTarget=(0.00714154, 0.0109552, 0.0715629), 
    viewOffsetX=0.00212315, viewOffsetY=0.00840877)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.162897, 
    farPlane=0.216955, width=0.0313082, height=0.0125016, cameraPosition=(
    -0.0332393, 0.0431651, -0.182544), cameraUpVector=(-0.0424626, -0.986754, 
    -0.156565), cameraTarget=(0.00818364, 0.00103783, 0.0717293), 
    viewOffsetX=0.00191371, viewOffsetY=0.0075793)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.175958, 
    farPlane=0.202345, width=0.0338185, height=0.013504, cameraPosition=(
    -0.0190461, 0.00132768, -0.188591), cameraUpVector=(-0.0392731, -0.997229, 
    0.0631752), cameraTarget=(0.00280148, 0.0169171, 0.0710723), 
    viewOffsetX=0.00206715, viewOffsetY=0.00818701)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.175593, 
    farPlane=0.20271, width=0.0337483, height=0.013476, cameraPosition=(
    -0.0192457, 0.00128414, -0.188572), cameraUpVector=(-0.0167549, -0.997977, 
    0.0613255), cameraTarget=(0.00260189, 0.0168736, 0.0710917), 
    viewOffsetX=0.00206286, viewOffsetY=0.00817001)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.175603, 
    farPlane=0.2027, width=0.0337503, height=0.0134768, cameraPosition=(
    -0.0194242, 0.00124115, -0.188554), cameraUpVector=(0.00349431, -0.998214, 
    0.059636), cameraTarget=(0.00242338, 0.0168306, 0.0711093), 
    viewOffsetX=0.00206298, viewOffsetY=0.00817048)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.153723, 
    farPlane=0.222958, width=0.029545, height=0.0117976, cameraPosition=(
    0.0255994, -0.0299323, -0.184457), cameraUpVector=(-0.0194418, -0.974644, 
    0.222917), cameraTarget=(-0.0143852, 0.0283412, 0.066841), 
    viewOffsetX=0.00180593, viewOffsetY=0.00715245)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.17271, 
    farPlane=0.205112, width=0.0331943, height=0.0132548, cameraPosition=(
    0.00105982, -0.00443618, -0.189228), cameraUpVector=(-0.0274926, -0.99568, 
    0.0886853), cameraTarget=(-0.00484912, 0.0188794, 0.0707077), 
    viewOffsetX=0.00202899, viewOffsetY=0.00803589)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.172179, 
    farPlane=0.205642, width=0.0330922, height=0.013214, cameraPosition=(
    0.000681861, -0.00452961, -0.189228), cameraUpVector=(0.0154422, -0.995851, 
    0.0896767), cameraTarget=(-0.00522708, 0.018786, 0.0707075), 
    viewOffsetX=0.00202275, viewOffsetY=0.00801118)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.172193, 
    farPlane=0.205628, width=0.033095, height=0.0132151, 
    viewOffsetX=0.00536356, viewOffsetY=0.00852011)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.171568, 
    farPlane=0.206253, width=0.039701, height=0.0158529, 
    viewOffsetX=0.00526068, viewOffsetY=0.00768877)
session.viewports['Viewport: 1'].view.setValues(session.views['Left'])
session.viewports['Viewport: 1'].view.setValues(session.views['Bottom'])
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0733876, 
    farPlane=0.326849, width=0.00712791, height=0.00284623, 
    viewOffsetX=-0.000597882, viewOffsetY=0.000112139)
o3 = session.odbs['C:/LocalUserData/User-data/nguyenb5/CP1000 plastic (UMAT UMATHT)/CP1000_dense_CSC/CP1000_NDBR40_dense_CSC/NDBR40_combined_processed.odb']
session.viewports['Viewport: 1'].setValues(displayedObject=o3)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.455582, 
    farPlane=0.630272, width=0.0682944, height=0.0272705, 
    viewOffsetX=-3.13282e-05, viewOffsetY=-0.0087785)
session.viewports['Viewport: 1'].odbDisplay.setValues(viewCut=ON)
session.viewports['Viewport: 1'].view.setValues(session.views['Back'])
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM1', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].odbDisplay.display.setValues(
    plotState=CONTOURS_ON_DEF)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.251515, 
    farPlane=0.260615, width=0.0295807, height=0.0118118, 
    viewOffsetX=-0.00152745, viewOffsetY=-0.0590882)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    motion=TRANSLATE)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.253085, 
    farPlane=0.259046, width=0.0160321, height=0.00640173, 
    viewOffsetX=-0.00171115, viewOffsetY=-0.0603751)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    position=0.0035205)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Cut-4'].setValues(
    showModelBelowCut=False)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Cut-4'].setValues(
    showModelOnCut=False)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    position=-0.00151952)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    position=-0.002)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    position=-0.001)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    position=-0.000)
session.viewports['Viewport: 1'].view.setValues(width=0.0170653, 
    height=0.00681432, viewOffsetX=-0.00134471, viewOffsetY=-0.060474)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    position=0.005)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    position=0.0005)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    motion=ROTATE)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    angle=182)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    motion=TRANSLATE)
session.viewports['Viewport: 1'].odbDisplay.ViewCut(name='Y-Plane-rotate', 
    objectToCopy=session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'])
session.viewports['Viewport: 1'].odbDisplay.setValues(viewCutNames=('Y-Plane', 
    'Y-Plane-rotate'), viewCut=ON)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane-rotate'].setValues(
    motion=ROTATE)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane-rotate'].setValues(
    angle=139)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.254471, 
    farPlane=0.257659, width=0.00707056, height=0.00282333, 
    viewOffsetX=-0.0006456, viewOffsetY=-0.060186)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane-rotate'].setValues(
    useCommonFreeBodyOptions=False)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane-rotate'].viewCutOptions.setValues(
    freeBodyCutThru=CURRENT_DISPLAY_GROUP, displayGroup='Current', 
    elementSet=' ALL ELEMENTS', freeBodyStepThru=ACTIVE_CUT_RANGE, 
    numCutFreeBody=1, cutFreeBodyMin=-0.100767, cutFreeBodyMax=0.0992572, 
    showHeatFlowRate=ON, displaySlicing=OFF, summationLoc=SPECIFY, 
    summationPoint=(0.0, 0.0, 0.003), yAxis=(0.0, 1.0, 0.0), 
    componentResolution=NORMAL_TANGENTIAL, csysName='(Global)')
session.viewports['Viewport: 1'].odbDisplay.setValues(viewCutNames=(
    'Y-Plane-rotate', ), viewCut=ON)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    position=0.0035205)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.252707, 
    farPlane=0.259423, width=0.0184728, height=0.00737633, 
    viewOffsetX=-0.00041806, viewOffsetY=-0.059561)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    position=0.00100049)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.255165, 
    farPlane=0.256965, width=0.00292647, height=0.00116856, 
    viewOffsetX=-0.000242291, viewOffsetY=-0.0603289)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane-rotate'].setValues(
    showModelBelowCut=False)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane-rotate'].setValues(
    showModelOnCut=False)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.254978, 
    farPlane=0.257152, width=0.00379389, height=0.00151493, 
    viewOffsetX=-0.000379721, viewOffsetY=-0.0602861)
session.viewports['Viewport: 1'].odbDisplay.setValues(viewCut=OFF)
session.viewports['Viewport: 1'].odbDisplay.setValues(viewCutNames=('Y-Plane', 
    ), viewCut=ON)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.25342, 
    farPlane=0.25871, width=0.015666, height=0.00625555, 
    viewOffsetX=-0.00142792, viewOffsetY=-0.0596282)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    useCommonFreeBodyOptions=False)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].viewCutOptions.setValues(
    freeBodyCutThru=CURRENT_DISPLAY_GROUP, displayGroup='Current', 
    elementSet=' ALL ELEMENTS', freeBodyStepThru=ACTIVE_CUT_RANGE, 
    numCutFreeBody=1, cutFreeBodyMin=-0.124997, cutFreeBodyMax=0.126998, 
    showHeatFlowRate=ON, displaySlicing=OFF, summationLoc=SPECIFY, 
    summationPoint=(0.0, 0.001, 0.0), yAxis=(0.0, 1.0, 0.0), 
    componentResolution=NORMAL_TANGENTIAL, csysName='(Global)')
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    motion=ROTATE)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    angle=0)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    origin=(0, 0.001, 0))
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    angle=155)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    angle=333)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    angle=3334)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    angle=334)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.252828, 
    farPlane=0.259302, width=0.0176883, height=0.00706309, 
    viewOffsetX=-0.000519584, viewOffsetY=-0.0592339)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.166252, 
    farPlane=0.347136, width=0.0116313, height=0.00464447, cameraPosition=(
    0.167508, 0.197441, -0.0535456), cameraUpVector=(0.104194, 0.45842, 
    0.882607), cameraTarget=(0.00637175, 0.029251, 0.0528337), 
    viewOffsetX=-0.000341662, viewOffsetY=-0.0389504)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.135819, 
    farPlane=0.377471, width=0.00950217, height=0.00379429, cameraPosition=(
    0.0669699, 0.255557, -0.0145706), cameraUpVector=(0.1094, 0.262837, 
    0.958618), cameraTarget=(0.006335, 0.0174806, 0.0576258), 
    viewOffsetX=-0.00027912, viewOffsetY=-0.0318205)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.205911, 
    farPlane=0.307056, width=0.0144059, height=0.00575239, cameraPosition=(
    -0.0759077, 0.150847, -0.202965), cameraUpVector=(0.185561, 0.929082, 
    0.319958), cameraTarget=(0.0106037, 0.0571058, 0.0190654), 
    viewOffsetX=-0.000423164, viewOffsetY=-0.048242)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.238659, 
    farPlane=0.274267, width=0.016697, height=0.00666723, cameraPosition=(
    -0.086376, 0.0855623, -0.234062), cameraUpVector=(-0.189393, 0.968332, 
    0.162674), cameraTarget=(-0.0119157, 0.0592064, 0.00951578), 
    viewOffsetX=-0.000490463, viewOffsetY=-0.0559143)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.159672, 
    farPlane=0.353434, width=0.0111709, height=0.00446062, cameraPosition=(
    0.00213087, 0.232328, -0.12617), cameraUpVector=(-0.0543165, 0.666159, 
    0.743829), cameraTarget=(-0.00370646, 0.041416, 0.0443809), 
    viewOffsetX=-0.000328138, viewOffsetY=-0.0374087)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.233478, 
    farPlane=0.279591, width=0.0163344, height=0.00652246, cameraPosition=(
    0.0128624, 0.102242, -0.242921), cameraUpVector=(0.0449378, 0.985197, 
    0.165429), cameraTarget=(0.0022685, 0.0603451, 0.00947096), 
    viewOffsetX=-0.000479814, viewOffsetY=-0.0547002)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.230971, 
    farPlane=0.282098, width=0.0267262, height=0.010672, 
    viewOffsetX=-0.000193825, viewOffsetY=-0.0555655)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.190651, 
    farPlane=0.321522, width=0.0220607, height=0.00880902, cameraPosition=(
    -0.231571, 0.123958, -0.0254834), cameraUpVector=(-0.0446775, 0.215132, 
    -0.975563), cameraTarget=(-0.00338784, 0.013124, -0.0603749), 
    viewOffsetX=-0.000159989, viewOffsetY=-0.0458656)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.209419, 
    farPlane=0.301511, width=0.0242323, height=0.00967618, cameraPosition=(
    -0.0692925, 0.132505, 0.216547), cameraUpVector=(-0.489407, 0.795272, 
    -0.357803), cameraTarget=(-0.0306829, 0.0487843, -0.0223445), 
    viewOffsetX=-0.000175738, viewOffsetY=-0.0503806)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.239036, 
    farPlane=0.271412, width=0.0276593, height=0.0110446, cameraPosition=(
    0.0906243, 0.0772654, 0.234085), cameraUpVector=(-0.2686, 0.961951, 
    0.0500534), cameraTarget=(-0.0174545, 0.0591287, 0.00265651), 
    viewOffsetX=-0.000200592, viewOffsetY=-0.0575057)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.248542, 
    farPlane=0.262292, width=0.0287593, height=0.0114838, cameraPosition=(
    0.0725151, 0.0617485, 0.244833), cameraUpVector=(-0.00163916, 0.999998, 
    -0.000654202), cameraTarget=(-0.000929639, 0.0614672, -0.000472901), 
    viewOffsetX=-0.000208569, viewOffsetY=-0.0597926)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.22765, 
    farPlane=0.283643, width=0.0263418, height=0.0105185, cameraPosition=(
    0.0565821, 0.0166664, 0.256058), cameraUpVector=(0.0338039, 0.985119, 
    0.168517), cameraTarget=(0.00129799, 0.0606655, 0.00993464), 
    viewOffsetX=-0.000191037, viewOffsetY=-0.0547664)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.245878, 
    farPlane=0.265337, width=0.0284509, height=0.0113607, cameraPosition=(
    0.00811715, 0.0719611, 0.252762), cameraUpVector=(0.0638793, 0.99704, 
    -0.0427931), cameraTarget=(0.00323466, 0.0612947, -0.00303369), 
    viewOffsetX=-0.000206333, viewOffsetY=-0.0591515)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.25031, 
    farPlane=0.260959, width=0.0289638, height=0.0115655, cameraPosition=(
    0.0116384, 0.0622346, 0.255182), cameraUpVector=(0.0711372, 0.99745, 
    -0.00571392), cameraTarget=(0.00367529, 0.061336, -0.000756961), 
    viewOffsetX=-0.000210052, viewOffsetY=-0.0602178)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.230486, 
    farPlane=0.281171, width=0.0266699, height=0.0106495, cameraPosition=(
    0.00174015, 0.0187683, 0.262276), cameraUpVector=(0.0485125, 0.985343, 
    0.163544), cameraTarget=(0.0022986, 0.0606682, 0.00966311), 
    viewOffsetX=-0.000193416, viewOffsetY=-0.0554487)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.242465, 
    farPlane=0.269124, width=0.0280561, height=0.011203, cameraPosition=(
    -0.0185011, 0.0443937, 0.258589), cameraUpVector=(0.0247825, 0.997349, 
    0.0684136), cameraTarget=(0.000855055, 0.0613881, 0.00382336), 
    viewOffsetX=-0.000203469, viewOffsetY=-0.0583306)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.232232, 
    farPlane=0.27936, width=0.026872, height=0.0107302, cameraPosition=(
    0.00522618, 0.0225694, 0.261917), cameraUpVector=(0.0218725, 0.988567, 
    0.149187), cameraTarget=(0.000651335, 0.0608725, 0.00877532), 
    viewOffsetX=-0.000194882, viewOffsetY=-0.0558688)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.232401, 
    farPlane=0.27919, width=0.0268915, height=0.010738, viewOffsetX=0.00192133, 
    viewOffsetY=-0.0576073)
session.viewports['Viewport: 1'].view.setValues(session.views['Top'])
session.viewports['Viewport: 1'].view.setValues(session.views['Bottom'])
session.viewports['Viewport: 1'].view.setValues(session.views['Back'])
session.viewports['Viewport: 1'].view.setValues(session.views['Bottom'])
session.viewports['Viewport: 1'].view.setValues(session.views['Top'])
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0697789, 
    farPlane=0.323797, width=0.00651035, height=0.00259963, 
    viewOffsetX=-0.000454189, viewOffsetY=0.000234425)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='SDV_AR48_RHO_D', outputPosition=INTEGRATION_POINT, )
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
session.viewports[session.currentViewportName].odbDisplay.setFrame(
    step='step2_insitu', frame=58)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0706091, 
    farPlane=0.323808, width=0.00658782, height=0.00263057, 
    viewOffsetX=-0.000706521, viewOffsetY=0.000270899)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0706694, 
    farPlane=0.323747, width=0.00659344, height=0.00263281, 
    viewOffsetX=-0.000678923, viewOffsetY=0.000186745)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.19132, 
    farPlane=0.201715, width=0.0178502, height=0.00712771, cameraPosition=(
    0.0193201, 0.00462188, 0.195525), cameraUpVector=(0.083619, 0.995978, 
    -0.0321832), cameraTarget=(-0.00591909, -0.00157282, -0.0617597), 
    viewOffsetX=-0.00183802, viewOffsetY=0.000505567)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.191101, 
    farPlane=0.202408, width=0.0178298, height=0.00711956, cameraPosition=(
    -0.00813058, 0.00351877, 0.196565), cameraUpVector=(0.0326702, 0.999331, 
    -0.0164522), cameraTarget=(0.00280397, -0.00109152, -0.0617561), 
    viewOffsetX=-0.00183592, viewOffsetY=0.000504989)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    origin=(0, 0.0005, 0))
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.190384, 
    farPlane=0.203124, width=0.0242033, height=0.00966458, 
    viewOffsetX=-0.00122576, viewOffsetY=-0.000200278)
session.viewports['Viewport: 1'].view.setValues(session.views['Front'])
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.247693, 
    farPlane=0.267526, width=0.0609654, height=0.024344, 
    viewOffsetX=0.00413631, viewOffsetY=0.0513821)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM1', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM5', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM8', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.131181, 
    farPlane=0.369471, width=0.032288, height=0.0128928, cameraPosition=(
    -0.0355724, 0.195819, 0.162075), cameraUpVector=(0.527612, 0.426201, 
    -0.734832), cameraTarget=(-0.0275452, -0.0293729, 0.0372276), 
    viewOffsetX=0.00219064, viewOffsetY=0.0272125)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.135374, 
    farPlane=0.365278, width=0.03332, height=0.0133049, cameraPosition=(
    -0.0106517, 0.191723, 0.171065), cameraUpVector=(0.0356329, 0.485536, 
    -0.87349), cameraTarget=(-0.00262449, -0.0334689, 0.0462179), 
    viewOffsetX=0.00226066, viewOffsetY=0.0280823)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.12481, 
    farPlane=0.369176, width=0.0307199, height=0.0122667, cameraPosition=(
    0.0308598, 0.249055, -0.03614), cameraUpVector=(0.159052, -0.372885, 
    -0.914144), cameraTarget=(-0.0103218, 0.0111689, 0.0537303), 
    viewOffsetX=0.00208425, viewOffsetY=0.0258909)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.118394, 
    farPlane=0.375146, width=0.0291407, height=0.0116361, cameraPosition=(
    -0.0423728, 0.249386, 0.0134531), cameraUpVector=(0.185308, -0.135408, 
    -0.973307), cameraTarget=(-0.00857576, -0.00260352, 0.0549453), 
    viewOffsetX=0.00197711, viewOffsetY=0.02456)
session.viewports['Viewport: 1'].view.setValues(session.views['Back'])
session.viewports['Viewport: 1'].view.setValues(session.views['Front'])
session.viewports['Viewport: 1'].view.setValues(session.views['Top'])
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.069485, 
    farPlane=0.322499, width=0.00683202, height=0.00272808, 
    viewOffsetX=3.38316e-05, viewOffsetY=-8.86415e-05)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.153766, 
    farPlane=0.23872, width=0.0151188, height=0.00603707, cameraPosition=(
    0.0392444, 0.0625024, 0.182035), cameraUpVector=(0.0652315, 0.940638, 
    -0.333085), cameraTarget=(-0.0123138, -0.0185624, -0.05699), 
    viewOffsetX=7.48673e-05, viewOffsetY=-0.000196158)
session.viewports['Viewport: 1'].view.setValues(session.views['Bottom'])
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.189964, 
    farPlane=0.448492, width=0.0238256, height=0.00951375, 
    viewOffsetX=-0.000646299, viewOffsetY=3.12692e-06)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.190281, 
    farPlane=0.448391, width=0.0238654, height=0.00952964, cameraPosition=(
    -0.0317854, -0.316813, 0.0150677), cameraUpVector=(-0.0923131, 0.0561244, 
    0.994147), cameraTarget=(-0.006143, -0.0607667, 0.00299373), 
    viewOffsetX=-0.000647379, viewOffsetY=3.13214e-06)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.214099, 
    farPlane=0.42386, width=0.0268526, height=0.0107225, cameraPosition=(
    0.0806551, -0.251526, -0.178003), cameraUpVector=(-0.152788, -0.602407, 
    0.78343), cameraTarget=(0.0155604, -0.0479627, -0.0341709), 
    viewOffsetX=-0.000728412, viewOffsetY=3.52419e-06)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.309561, 
    farPlane=0.327665, width=0.0388255, height=0.0155034, cameraPosition=(
    0.173112, -0.00120641, -0.267475), cameraUpVector=(0.027886, -0.999318, 
    0.0242124), cameraTarget=(0.033246, 0.000131932, -0.0511453), 
    viewOffsetX=-0.00105319, viewOffsetY=5.09554e-06)
session.viewports['Viewport: 1'].view.setValues(session.views['Top'])
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0702931, 
    farPlane=0.321691, width=0.00571717, height=0.00228291, cameraPosition=(
    -0.000769672, 0.195352, 0.0218252), cameraUpVector=(-0.0403687, 0.11111, 
    -0.992988), cameraTarget=(0.000241977, -0.060654, -0.00686164))
session.viewports['Viewport: 1'].view.setValues(session.views['Top'])
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0695332, 
    farPlane=0.322451, width=0.00642655, height=0.00256617, 
    viewOffsetX=-0.000537181, viewOffsetY=0.000408114)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM5', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM1', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.174928, 
    farPlane=0.214305, width=0.0161676, height=0.00645583, cameraPosition=(
    0.0326393, -0.0269649, 0.189876), cameraUpVector=(-0.0764863, 0.985754, 
    0.149794), cameraTarget=(-0.0101794, 0.00794801, -0.0617393), 
    viewOffsetX=-0.00135141, viewOffsetY=0.00102671)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.175815, 
    farPlane=0.213864, width=0.0162496, height=0.00648856, cameraPosition=(
    -0.00131817, -0.0257491, 0.193053), cameraUpVector=(0.016472, 0.991578, 
    0.128456), cameraTarget=(0.000704571, 0.00731292, -0.0624185), 
    viewOffsetX=-0.00135826, viewOffsetY=0.00103191)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.175326, 
    farPlane=0.214354, width=0.020755, height=0.00828762, 
    viewOffsetX=-0.00114233, viewOffsetY=0.000982449)
session.viewports['Viewport: 1'].view.setValues(session.views['Front'])
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.249237, 
    farPlane=0.265983, width=0.0509524, height=0.0203457, 
    viewOffsetX=0.00190783, viewOffsetY=0.0525522)
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.251945, 
    farPlane=0.263274, width=0.0334006, height=0.0133371, 
    viewOffsetX=0.00207966, viewOffsetY=0.0568799)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.245585, 
    farPlane=0.269522, width=0.0325574, height=0.0130004, cameraPosition=(
    0.010592, -0.047992, 0.260083), cameraUpVector=(0.0907128, 0.994407, 
    -0.0540826), cameraTarget=(-0.00526292, -0.0605131, 0.00326614), 
    viewOffsetX=0.00202716, viewOffsetY=0.055444)
session.viewports['Viewport: 1'].odbDisplay.setValues(viewCut=OFF)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.222758, 
    farPlane=0.291178, width=0.0295312, height=0.011792, cameraPosition=(
    0.0360705, -0.00181669, 0.261909), cameraUpVector=(-0.149268, 0.966958, 
    -0.206668), cameraTarget=(0.00871169, -0.059387, 0.0123085), 
    viewOffsetX=0.00183874, viewOffsetY=0.0502905)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.223202, 
    farPlane=0.290736, width=0.02959, height=0.0118155, cameraPosition=(
    0.029815, -0.00203375, 0.262645), cameraUpVector=(-0.0418207, 0.974558, 
    -0.220198), cameraTarget=(0.00245623, -0.0596041, 0.0130442), 
    viewOffsetX=0.0018424, viewOffsetY=0.0503906)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    origin=(0, 0.001, 0))
session.viewports['Viewport: 1'].odbDisplay.setValues(viewCut=ON)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.222322, 
    farPlane=0.291616, width=0.0334924, height=0.0133738, 
    viewOffsetX=0.000882509, viewOffsetY=0.0508812)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM5', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.222912, 
    farPlane=0.291026, width=0.0296724, height=0.0118484, 
    viewOffsetX=0.00124586, viewOffsetY=0.0522174)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.243517, 
    farPlane=0.270933, width=0.0324152, height=0.0129436, cameraPosition=(
    0.0206975, -0.0439983, 0.259901), cameraUpVector=(0.0194923, 0.997549, 
    -0.0672084), cameraTarget=(-0.00122488, -0.0608258, 0.00377669), 
    viewOffsetX=0.00136103, viewOffsetY=0.0570442)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.210613, 
    farPlane=0.303025, width=0.0280353, height=0.0111947, cameraPosition=(
    0.02591, 0.0244013, 0.261842), cameraUpVector=(0.0294997, 0.945722, 
    -0.323636), cameraTarget=(-0.0018331, -0.0577469, 0.0192608), 
    viewOffsetX=0.00117713, viewOffsetY=0.0493364)
session.viewports['Viewport: 1'].view.setValues(session.views['Front'])
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.248526, 
    farPlane=0.268662, width=0.0619478, height=0.0247362, 
    viewOffsetX=0.00369017, viewOffsetY=0.0507737)
session.viewports['Viewport: 1'].view.setValues(session.views['Top'])
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0695764, 
    farPlane=0.324, width=0.00884511, height=0.00353192, 
    viewOffsetX=-0.000585845, viewOffsetY=0.000470567)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.166614, 
    farPlane=0.224727, width=0.0211813, height=0.00845786, cameraPosition=(
    -0.0228807, 0.0407616, 0.190224), cameraUpVector=(0.112835, 0.973982, 
    -0.196539), cameraTarget=(0.007133, -0.0133797, -0.0608518), 
    viewOffsetX=-0.00140292, viewOffsetY=0.00112686)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.183893, 
    farPlane=0.205619, width=0.0233779, height=0.00933498, cameraPosition=(
    0.0823768, -0.00662577, 0.176307), cameraUpVector=(0.169306, 0.984568, 
    -0.044283), cameraTarget=(-0.0270252, 0.00165497, -0.057859), 
    viewOffsetX=-0.00154841, viewOffsetY=0.00124372)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.155754, 
    farPlane=0.235164, width=0.0198006, height=0.00790655, cameraPosition=(
    -0.0143792, 0.0542604, 0.18752), cameraUpVector=(0.110748, 0.956258, 
    -0.270749), cameraTarget=(0.00501216, -0.0180652, -0.059996), 
    viewOffsetX=-0.00131147, viewOffsetY=0.00105341)
o3 = session.odbs['C:/LocalUserData/User-data/nguyenb5/CP1000 plastic (UMAT UMATHT)/CP1000_dense_CSC/CP1000_SH115_dense_CSC/SH115_combined_processed.odb']
session.viewports['Viewport: 1'].setValues(displayedObject=o3)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.454847, 
    farPlane=0.631006, width=0.0824278, height=0.0329141, 
    viewOffsetX=0.00169849, viewOffsetY=-0.0106886)
session.viewports['Viewport: 1'].view.setValues(session.views['Front'])
session.viewports['Viewport: 1'].view.setValues(session.views['Back'])
session.viewports['Viewport: 1'].view.setValues(session.views['Front'])
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.489014, 
    farPlane=0.512792, width=0.0736713, height=0.0294175, 
    viewOffsetX=-0.00185167, viewOffsetY=-8.6613e-08)
session.viewports['Viewport: 1'].view.setValues(session.views['Back'])
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.480878, 
    farPlane=0.520929, width=0.126433, height=0.0504857, 
    viewOffsetX=0.000883631, viewOffsetY=0.0084168)
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.488971, 
    farPlane=0.512835, width=0.0736648, height=0.0294149, 
    viewOffsetX=0.00169266, viewOffsetY=0.00377913)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='SDV_AR48_RHO_D', outputPosition=INTEGRATION_POINT, )
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
session.viewports[session.currentViewportName].odbDisplay.setFrame(
    step='step2_insitu', frame=92)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM1', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.495031, 
    farPlane=0.506775, width=0.0343506, height=0.0137164, 
    viewOffsetX=0.000464342, viewOffsetY=-0.000632735)
session.viewports['Viewport: 1'].odbDisplay.setValues(viewCut=ON)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    angle=80)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.497767, 
    farPlane=0.504039, width=0.0188441, height=0.0075246, 
    viewOffsetX=-0.000389334, viewOffsetY=0.00113064)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    angle=78)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.490707, 
    farPlane=0.5111, width=0.0703835, height=0.0281047, viewOffsetX=0.00109003, 
    viewOffsetY=0.000698609)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.490116, 
    farPlane=0.51169, width=0.0702987, height=0.0280708, cameraPosition=(
    -2.74779e-05, -4.40215e-05, -0.500903), cameraUpVector=(0.0370403, 
    0.999314, 0), cameraTarget=(-2.74779e-05, -4.40215e-05, 0), 
    viewOffsetX=0.00108872, viewOffsetY=0.000697768)
session.viewports['Viewport: 1'].view.setValues(session.views['Front'])
session.viewports['Viewport: 1'].view.setValues(session.views['Back'])
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.318703, 
    farPlane=0.331415, width=0.037413, height=0.0149393, 
    viewOffsetX=0.000512908, viewOffsetY=0.0419352)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM5', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.286247, 
    farPlane=0.363611, width=0.033603, height=0.0134179, cameraPosition=(
    -0.269481, 0.0296861, -0.184816), cameraUpVector=(0.400518, 0.893842, 
    -0.201571), cameraTarget=(-0.0173321, -0.0397677, 0.00821401), 
    viewOffsetX=0.000460674, viewOffsetY=0.0376646)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.273987, 
    farPlane=0.375562, width=0.0321638, height=0.0128432, cameraPosition=(
    -0.314049, 0.0578459, -0.0751338), cameraUpVector=(0.275518, 0.94697, 
    0.16534), cameraTarget=(-0.0120935, -0.0417947, -0.00762019), 
    viewOffsetX=0.000440944, viewOffsetY=0.0360514)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.312544, 
    farPlane=0.338454, width=0.0366901, height=0.0146506, cameraPosition=(
    -0.325233, -0.0415747, -0.019328), cameraUpVector=(0.0245655, 0.9728, 
    -0.230339), cameraTarget=(-0.00143526, -0.0429839, 0.00925205), 
    viewOffsetX=0.000502997, viewOffsetY=0.0411248)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.311886, 
    farPlane=0.339114, width=0.0366129, height=0.0146198, cameraPosition=(
    -0.327925, -0.0101216, 0.0127171), cameraUpVector=(0.0865709, 0.229347, 
    -0.969487), cameraTarget=(-0.00412684, -0.0115308, 0.0412971), 
    viewOffsetX=0.000501938, viewOffsetY=0.0410382)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.274215, 
    farPlane=0.375462, width=0.0321907, height=0.012854, cameraPosition=(
    -0.304194, 0.100316, 0.0702085), cameraUpVector=(-0.118963, -0.0941437, 
    -0.988425), cameraTarget=(0.00453649, 0.00246744, 0.04237), 
    viewOffsetX=0.000441312, viewOffsetY=0.0360815)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.3045, 
    farPlane=0.346111, width=0.0357459, height=0.0142736, cameraPosition=(
    -0.327202, 0.0226434, -0.0110785), cameraUpVector=(0.160483, -0.0559378, 
    -0.985452), cameraTarget=(-0.00728812, 0.000861519, 0.0422559), 
    viewOffsetX=0.000490051, viewOffsetY=0.0400664)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.307459, 
    farPlane=0.343151, width=0.0126067, height=0.00503394, 
    viewOffsetX=0.0020983, viewOffsetY=0.0406647)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.303201, 
    farPlane=0.347354, width=0.012432, height=0.00496421, cameraPosition=(
    -0.326344, 0.0302013, 0.0168165), cameraUpVector=(0.0812288, 0.0150413, 
    -0.996582), cameraTarget=(-0.00393088, -0.00218153, 0.0426061), 
    viewOffsetX=0.00206924, viewOffsetY=0.0401014)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.293014, 
    farPlane=0.357497, width=0.0120143, height=0.00479742, cameraPosition=(
    -0.323179, 0.0569418, 0.00685258), cameraUpVector=(0.113414, 0.0113179, 
    -0.993483), cameraTarget=(-0.00549161, -0.00197798, 0.0424471), 
    viewOffsetX=0.00199972, viewOffsetY=0.0387541)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.289382, 
    farPlane=0.361126, width=0.0118654, height=0.00473795, cameraPosition=(
    -0.320624, 0.0698881, -0.00852177), cameraUpVector=(0.14609, -0.0635364, 
    -0.987229), cameraTarget=(-0.00698256, 0.00126566, 0.0423067), 
    viewOffsetX=0.00197493, viewOffsetY=0.0382737)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.303576, 
    farPlane=0.346992, width=0.0124474, height=0.00497034, cameraPosition=(
    -0.326571, 0.0318605, 0.00615253), cameraUpVector=(0.108707, -0.0388277, 
    -0.993315), cameraTarget=(-0.00511926, 0.000138298, 0.0425712), 
    viewOffsetX=0.0020718, viewOffsetY=0.0401509)
session.viewports['Viewport: 1'].view.setValues(session.views['Back'])
session.viewports['Viewport: 1'].view.setValues(session.views['Front'])
session.viewports['Viewport: 1'].view.setValues(session.views['Back'])
session.viewports['Viewport: 1'].view.setValues(session.views['Bottom'])
session.viewports['Viewport: 1'].view.setValues(session.views['Top'])
session.viewports['Viewport: 1'].view.setValues(session.views['Top'])
session.viewports['Viewport: 1'].view.setValues(session.views['Bottom'])
session.viewports['Viewport: 1'].view.setValues(session.views['Left'])
session.viewports['Viewport: 1'].view.setValues(session.views['Right'])
session.viewports['Viewport: 1'].view.setValues(session.views['Left'])
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.311857, 
    farPlane=0.338157, width=0.0351826, height=0.0140487, 
    viewOffsetX=0.000903025, viewOffsetY=0.0439237)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.303439, 
    farPlane=0.338897, width=0.0342329, height=0.0136694, cameraUpVector=(
    0.0240517, 0.997585, -0.0651635), cameraTarget=(-0.00368424, -0.0566543, 
    0.0343842), viewOffsetX=0.000878649, viewOffsetY=0.0427381)
session.viewports['Viewport: 1'].view.setValues(session.views['Left'])
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.313958, 
    farPlane=0.336056, width=0.0215907, height=0.00862131, 
    viewOffsetX=0.000675388, viewOffsetY=0.0431753)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.314156, 
    farPlane=0.335858, width=0.0216043, height=0.00862676, cameraPosition=(
    -0.324992, -0.0434196, 0.00851113), cameraUpVector=(0, 0.981757, 
    -0.190142), cameraTarget=(6.66976e-05, -0.0434196, 0.00851113), 
    viewOffsetX=0.000675814, viewOffsetY=0.0432026)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.226531, 
    farPlane=0.42237, width=0.0155784, height=0.00622056, cameraPosition=(
    -0.222507, 0.219529, -0.100189), cameraUpVector=(0.754028, 0.290843, 
    -0.588942), cameraTarget=(-0.0338713, -0.0128878, 0.026547), 
    viewOffsetX=0.000487314, viewOffsetY=0.0311524)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.226643, 
    farPlane=0.422521, width=0.0155861, height=0.00622362, cameraPosition=(
    -0.224995, 0.238274, -0.0215406), cameraUpVector=(0.177879, -0.11057, 
    -0.977821), cameraTarget=(-0.00840292, 0.00498475, 0.0442405), 
    viewOffsetX=0.000487554, viewOffsetY=0.0311677)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.308151, 
    farPlane=0.341586, width=0.0211914, height=0.00846183, cameraPosition=(
    -0.325138, -0.016195, 0.0395131), cameraUpVector=(0.0186471, -0.0224251, 
    -0.999575), cameraTarget=(-0.000594413, 0.00120094, 0.0451774), 
    viewOffsetX=0.000662894, viewOffsetY=0.0423766)
session.viewports['Viewport: 1'].view.setValues(session.views['User-1'])
session.viewports['Viewport: 1'].view.setValues(session.views['User-1'])
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.303102, 
    farPlane=0.347016, width=0.138807, height=0.0554268, 
    viewOffsetX=0.00114698, viewOffsetY=0.0300971)
session.viewports['Viewport: 1'].view.setValues(session.views['Back'])
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.319681, 
    farPlane=0.330437, width=0.03117, height=0.0124464, 
    viewOffsetX=-0.00202369, viewOffsetY=0.043337)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.319962, 
    farPlane=0.330156, width=0.0311974, height=0.0124574, cameraPosition=(
    0.0139954, -0.0425228, -0.325059), cameraUpVector=(-0.313729, 0.949513, 0), 
    cameraTarget=(0.0139954, -0.0425228, 0), viewOffsetX=-0.00202547, 
    viewOffsetY=0.043375)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.308271, 
    farPlane=0.339272, width=0.0300575, height=0.0120022, cameraPosition=(
    -0.165938, -0.0175117, -0.281174), cameraUpVector=(-0.545533, 0.714281, 
    0.438402), cameraTarget=(0.0249334, -0.0329483, -0.0185088), 
    viewOffsetX=-0.00195146, viewOffsetY=0.0417902)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.286659, 
    farPlane=0.360999, width=0.0279502, height=0.0111607, cameraPosition=(
    -0.189803, -0.102291, -0.245522), cameraUpVector=(-0.611487, 0.702314, 
    0.36447), cameraTarget=(0.0283022, -0.0320428, -0.0149608), 
    viewOffsetX=-0.00181465, viewOffsetY=0.0388604)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.31202, 
    farPlane=0.335515, width=0.030423, height=0.0121481, cameraPosition=(
    -0.159448, -0.00035875, -0.285489), cameraUpVector=(-0.790908, 0.110995, 
    0.601785), cameraTarget=(0.0367488, -0.00702172, -0.0264036), 
    viewOffsetX=-0.0019752, viewOffsetY=0.0422985)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.302444, 
    farPlane=0.3429, width=0.0294893, height=0.0117753, cameraPosition=(
    -0.30396, -0.0266713, -0.114112), cameraUpVector=(-0.222733, 0.0807874, 
    0.971526), cameraTarget=(0.0125886, -0.00558777, -0.0432927), 
    viewOffsetX=-0.00191458, viewOffsetY=0.0410004)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.301286, 
    farPlane=0.343609, width=0.0293764, height=0.0117302, cameraPosition=(
    -0.308737, 0.0197178, -0.101783), cameraUpVector=(-0.175193, 0.0376161, 
    0.983815), cameraTarget=(0.0102739, -0.00403256, -0.0440674), 
    viewOffsetX=-0.00190725, viewOffsetY=0.0408434)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.294164, 
    farPlane=0.350175, width=0.028682, height=0.0114529, cameraPosition=(
    -0.317046, 0.044241, -0.0586872), cameraUpVector=(-0.0575577, -0.121079, 
    0.990973), cameraTarget=(0.00506952, 0.00278702, -0.0450434), 
    viewOffsetX=-0.00186217, viewOffsetY=0.039878)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.198056, 
    farPlane=0.447997, width=0.0193111, height=0.00771106, cameraPosition=(
    -0.0766477, -0.315944, 0.00824442), cameraUpVector=(-0.563747, 0.29906, 
    0.769905), cameraTarget=(0.027037, -0.0108247, -0.0343548), 
    viewOffsetX=-0.00125377, viewOffsetY=0.0268492)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.243043, 
    farPlane=0.401006, width=0.0236974, height=0.00946256, cameraPosition=(
    0.15702, -0.173932, 0.224527), cameraUpVector=(-0.932664, -0.249108, 
    0.260927), cameraTarget=(0.0401923, 0.0143879, -0.0132757), 
    viewOffsetX=-0.00153855, viewOffsetY=0.0329478)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.294194, 
    farPlane=0.348019, width=0.0286848, height=0.0114541, cameraPosition=(
    -0.0339891, -0.0626603, 0.31601), cameraUpVector=(-0.949367, 0.248557, 
    -0.192151), cameraTarget=(0.0432152, -0.00822458, 0.00498026), 
    viewOffsetX=-0.00186236, viewOffsetY=0.0398821)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.307066, 
    farPlane=0.334399, width=0.0299399, height=0.0119553, cameraPosition=(
    -0.198961, -0.00775639, 0.255201), cameraUpVector=(-0.698179, 0.0148269, 
    -0.71577), cameraTarget=(0.0337331, 0.00149277, 0.0284167), 
    viewOffsetX=-0.00194384, viewOffsetY=0.0416271)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.306883, 
    farPlane=0.334582, width=0.0299221, height=0.0119482, cameraPosition=(
    -0.203976, -0.0342529, 0.248975), cameraUpVector=(-0.563084, 0.614374, 
    -0.552703), cameraTarget=(0.0287179, -0.0250038, 0.0221902), 
    viewOffsetX=-0.00194268, viewOffsetY=0.0416023)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.254269, 
    farPlane=0.385809, width=0.0247921, height=0.00989974, cameraPosition=(
    -0.294832, 0.127098, 0.0390286), cameraUpVector=(0.155042, 0.304602, 
    -0.939776), cameraTarget=(-0.00172088, -0.0134059, 0.0418443), 
    viewOffsetX=-0.00160962, viewOffsetY=0.0344698)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.302013, 
    farPlane=0.338591, width=0.0294473, height=0.0117586, cameraPosition=(
    -0.32265, 0.0208992, -0.000462552), cameraUpVector=(0.135755, -0.0204233, 
    -0.990532), cameraTarget=(-0.00114641, 0.00289028, 0.0439712), 
    viewOffsetX=-0.00191186, viewOffsetY=0.0409422)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.3029, 
    farPlane=0.337672, width=0.0295337, height=0.0117931, cameraPosition=(
    -0.321904, 0.0178371, -0.0241329), cameraUpVector=(0.207094, -0.0393468, 
    -0.97753), cameraTarget=(-0.00431586, 0.00376972, 0.0437152), 
    viewOffsetX=-0.00191747, viewOffsetY=0.0410624)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.30785, 
    farPlane=0.332809, width=0.0300163, height=0.0119858, cameraPosition=(
    -0.322849, 0.00282356, -0.0167659), cameraUpVector=(0.186198, -0.0422463, 
    -0.981604), cameraTarget=(-0.00347682, 0.00412291, 0.0437587), 
    viewOffsetX=-0.00194881, viewOffsetY=0.0417334)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.297924, 
    farPlane=0.34289, width=0.0290485, height=0.0115993, cameraPosition=(
    -0.322587, -0.0213952, 0.00072912), cameraUpVector=(0.137844, -0.0657037, 
    -0.988272), cameraTarget=(-0.00150841, 0.00550453, 0.0437242), 
    viewOffsetX=-0.00188597, viewOffsetY=0.0403878)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.298063, 
    farPlane=0.342751, width=0.0290621, height=0.0116047, cameraPosition=(
    -0.322333, -0.0248947, 0.00101922), cameraUpVector=(0.131479, 0.0149588, 
    -0.991206), cameraTarget=(-0.00125408, 0.00200507, 0.0440143), 
    viewOffsetX=-0.00188685, viewOffsetY=0.0404066)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.299856, 
    farPlane=0.340959, width=0.0157475, height=0.00628808, 
    viewOffsetX=-0.00122438, viewOffsetY=0.0413962)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM1', outputPosition=INTEGRATION_POINT, )
session.viewports[session.currentViewportName].odbDisplay.setFrame(
    step='step2_insitu', frame=24)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.245306, 
    farPlane=0.393859, width=0.0128827, height=0.00514416, cameraPosition=(
    -0.040061, -0.188749, 0.258343), cameraUpVector=(-0.951814, 0.287119, 
    -0.107762), cameraTarget=(0.0435318, -0.00808, 0.00137088), 
    viewOffsetX=-0.00100165, viewOffsetY=0.0338655)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.265772, 
    farPlane=0.373938, width=0.0139575, height=0.00557333, cameraPosition=(
    -0.0592109, 0.164107, 0.271871), cameraUpVector=(-0.539722, -0.81501, 
    0.210851), cameraTarget=(0.0240809, 0.0348555, -0.0145212), 
    viewOffsetX=-0.00108522, viewOffsetY=0.0366909)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.290507, 
    farPlane=0.349035, width=0.0152565, height=0.00609202, cameraPosition=(
    -0.056074, -0.0254256, 0.316871), cameraUpVector=(-0.451881, -0.844317, 
    -0.287979), cameraTarget=(0.0200097, 0.0393557, 0.00755118), 
    viewOffsetX=-0.00118622, viewOffsetY=0.0401056)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.276526, 
    farPlane=0.363012, width=0.0145223, height=0.00579883, cameraPosition=(
    0.244217, 0.126893, 0.168922), cameraUpVector=(-0.131023, -0.826638, 
    0.547268), cameraTarget=(0.000923555, 0.0360309, -0.0265686), 
    viewOffsetX=-0.00112913, viewOffsetY=0.0381755)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.284843, 
    farPlane=0.354829, width=0.0149591, height=0.00597324, cameraPosition=(
    -0.165695, 0.111235, 0.253935), cameraUpVector=(0.318154, -0.856037, 
    0.407405), cameraTarget=(-0.0126799, 0.0363468, -0.0229091), 
    viewOffsetX=-0.00116309, viewOffsetY=0.0393237)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.280187, 
    farPlane=0.359342, width=0.0147146, height=0.0058756, cameraPosition=(
    0.0584152, -0.10162, 0.300681), cameraUpVector=(0.930666, 0.28321, 
    -0.231631), cameraTarget=(-0.0424788, -0.0119056, 0.00498454), 
    viewOffsetX=-0.00114408, viewOffsetY=0.0386809)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.225606, 
    farPlane=0.414071, width=0.0118482, height=0.00473103, cameraPosition=(
    -0.208022, 0.200899, 0.14391), cameraUpVector=(0.836276, 0.517741, 
    0.180519), cameraTarget=(-0.0339306, -0.0268739, -0.00931079), 
    viewOffsetX=-0.000921212, viewOffsetY=0.0311458)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.260217, 
    farPlane=0.379442, width=0.0136659, height=0.00545683, cameraPosition=(
    -0.199014, 0.122018, 0.223166), cameraUpVector=(0.860784, 0.321575, 
    0.394514), cameraTarget=(-0.0352378, -0.0172518, -0.0206496), 
    viewOffsetX=-0.00106254, viewOffsetY=0.035924)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.259952, 
    farPlane=0.379706, width=0.013652, height=0.00545128, cameraPosition=(
    -0.184906, 0.100456, 0.244959), cameraUpVector=(0.558627, 0.823915, 
    -0.0953926), cameraTarget=(-0.0211303, -0.0388137, 0.00114307), 
    viewOffsetX=-0.00106146, viewOffsetY=0.0358874)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.296945, 
    farPlane=0.342737, width=0.0155948, height=0.00622703, cameraPosition=(
    0.0325326, 0.0103333, 0.32107), cameraUpVector=(0.449077, 0.868431, 
    -0.210136), cameraTarget=(-0.0194791, -0.0394705, 0.00408536), 
    viewOffsetX=-0.00121251, viewOffsetY=0.0409944)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.281709, 
    farPlane=0.357926, width=0.0147946, height=0.00590752, cameraPosition=(
    -0.285506, 0.0315887, 0.147418), cameraUpVector=(0.247545, 0.968536, 
    -0.0256726), cameraTarget=(-0.00581955, -0.0438036, -8.27815e-05), 
    viewOffsetX=-0.0011503, viewOffsetY=0.038891)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.235562, 
    farPlane=0.404115, width=0.0123711, height=0.00493981, cameraPosition=(
    -0.117906, 0.168607, -0.248996), cameraUpVector=(0.292281, 0.773798, 
    0.561969), cameraTarget=(-0.0123606, -0.0371611, -0.0205567), 
    viewOffsetX=-0.00096187, viewOffsetY=0.0325203)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.276727, 
    farPlane=0.362932, width=0.014533, height=0.00580305, cameraPosition=(
    -0.0600235, 0.0647143, -0.310613), cameraUpVector=(0.45467, 0.859753, 
    0.232594), cameraTarget=(-0.0205704, -0.0388224, -0.00501777), 
    viewOffsetX=-0.00112996, viewOffsetY=0.0382033)
session.viewports['Viewport: 1'].odbDisplay.setValues(viewCut=OFF)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.274577, 
    farPlane=0.365082, width=0.0288491, height=0.0115197, 
    viewOffsetX=-0.000815094, viewOffsetY=0.0381678)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.307831, 
    farPlane=0.331537, width=0.032343, height=0.0129148, cameraPosition=(
    -0.00400163, -0.0249411, -0.321697), cameraUpVector=(0.167689, 0.984146, 
    0.0577627), cameraTarget=(-0.00861928, -0.0431994, 0.00281853), 
    viewOffsetX=-0.000913809, viewOffsetY=0.0427903)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.307331, 
    farPlane=0.332038, width=0.0322904, height=0.0128938, cameraPosition=(
    0.0167408, -0.0240058, -0.321349), cameraUpVector=(-0.298412, 0.953158, 
    0.0493869), cameraTarget=(0.0121231, -0.0422641, 0.00316631), 
    viewOffsetX=-0.000912324, viewOffsetY=0.0427208)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.286906, 
    farPlane=0.352625, width=0.0301444, height=0.0120369, cameraPosition=(
    -0.0126072, 0.0275077, -0.32138), cameraUpVector=(-0.207091, 0.951917, 
    0.22576), cameraTarget=(0.00846929, -0.0429964, -0.0047572), 
    viewOffsetX=-0.000851692, viewOffsetY=0.0398816)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.289625, 
    farPlane=0.349906, width=0.0127966, height=0.00510977, 
    viewOffsetX=-0.000999089, viewOffsetY=0.0403551)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.305168, 
    farPlane=0.334513, width=0.0134833, height=0.00538399, cameraPosition=(
    -0.24358, -0.0600653, 0.203132), cameraUpVector=(0.109865, 0.972786, 
    0.204), cameraTarget=(-0.000233205, -0.0425013, -0.0116678), 
    viewOffsetX=-0.00105271, viewOffsetY=0.0425208)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.302674, 
    farPlane=0.337267, width=0.0133731, height=0.00533999, cameraPosition=(
    0.156819, -0.0150425, 0.281949), cameraUpVector=(0.0219346, 0.993189, 
    -0.114429), cameraTarget=(-0.0025434, -0.043992, 0.000114254), 
    viewOffsetX=-0.00104411, viewOffsetY=0.0421733)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.301414, 
    farPlane=0.338528, width=0.0235418, height=0.00940043, 
    viewOffsetX=-0.00210812, viewOffsetY=0.0427959)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.255764, 
    farPlane=0.386885, width=0.0199764, height=0.00797672, cameraPosition=(
    -0.101474, 0.113606, 0.286499), cameraUpVector=(0.161723, 0.880217, 
    -0.446165), cameraTarget=(-0.00472682, -0.040657, 0.0172253), 
    viewOffsetX=-0.00178884, viewOffsetY=0.0363144)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM8', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.260414, 
    farPlane=0.384038, width=0.0203396, height=0.00812174, cameraPosition=(
    -0.306203, 0.109474, -0.0156789), cameraUpVector=(0.291392, 0.512536, 
    0.807712), cameraTarget=(-0.011156, -0.0253458, -0.0365685), 
    viewOffsetX=-0.00182136, viewOffsetY=0.0369746)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.306043, 
    farPlane=0.339537, width=0.0239035, height=0.00954481, cameraPosition=(
    -0.102414, -0.072625, -0.300694), cameraUpVector=(-0.00194302, 0.995835, 
    -0.0911532), cameraTarget=(-0.000884501, -0.0442789, 0.00680148), 
    viewOffsetX=-0.00214049, viewOffsetY=0.0434532)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.300489, 
    farPlane=0.345654, width=0.0234697, height=0.0093716, cameraPosition=(
    -0.0423056, 0.000994636, -0.323446), cameraUpVector=(-0.0610436, 0.986869, 
    0.149547), cameraTarget=(0.00133122, -0.0446256, -0.00457338), 
    viewOffsetX=-0.00210165, viewOffsetY=0.0426646)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.303649, 
    farPlane=0.342494, width=0.00312502, height=0.00124784, 
    viewOffsetX=-0.00146797, viewOffsetY=0.0435593)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.314281, 
    farPlane=0.33153, width=0.00323443, height=0.00129153, cameraPosition=(
    0.033283, -0.0271286, -0.323181), cameraUpVector=(0.0113151, 0.998363, 
    0.0560583), cameraTarget=(-0.00251615, -0.0448354, -0.000582084), 
    viewOffsetX=-0.00151937, viewOffsetY=0.0450844)
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=25 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=26 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=25 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=24 )
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.313264, 
    farPlane=0.332547, width=0.00950657, height=0.00379605, 
    viewOffsetX=-0.00200728, viewOffsetY=0.044032)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.298701, 
    farPlane=0.347219, width=0.00906462, height=0.00361957, cameraPosition=(
    0.0544581, 0.0101754, -0.321364), cameraUpVector=(0.00409788, 0.985196, 
    0.171383), cameraTarget=(-0.00230858, -0.0444485, -0.00599264), 
    viewOffsetX=-0.00191396, viewOffsetY=0.041985)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.296396, 
    farPlane=0.349524, width=0.0238447, height=0.00952138, 
    viewOffsetX=-0.00261248, viewOffsetY=0.0426071)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.233028, 
    farPlane=0.416283, width=0.0187468, height=0.00748576, cameraPosition=(
    0.226855, 0.18843, 0.143779), cameraUpVector=(-0.130008, 0.544387, 
    -0.828698), cameraTarget=(0.00774013, -0.0265859, 0.0369045), 
    viewOffsetX=-0.00205394, viewOffsetY=0.033498)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.307502, 
    farPlane=0.341228, width=0.0247381, height=0.00987815, cameraPosition=(
    -0.0386565, -0.0124284, 0.325135), cameraUpVector=(0.169067, 0.982003, 
    -0.0841785), cameraTarget=(-0.00517735, -0.0457596, 0.00352461), 
    viewOffsetX=-0.00271036, viewOffsetY=0.0442037)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.30663, 
    farPlane=0.342099, width=0.0246679, height=0.00985012, cameraPosition=(
    -0.0611528, 0.0709178, 0.314155), cameraUpVector=(0.540221, -0.829423, 
    0.142191), cameraTarget=(-0.0276737, 0.0375867, -0.0074551), 
    viewOffsetX=-0.00270267, viewOffsetY=0.0440783)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.216855, 
    farPlane=0.431188, width=0.0174456, height=0.00696621, cameraPosition=(
    0.177769, -0.272686, 0.0327007), cameraUpVector=(0.410318, 0.336255, 
    -0.847686), cameraTarget=(-0.0216573, -0.0160422, 0.0379714), 
    viewOffsetX=-0.00191138, viewOffsetY=0.0311731)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.276705, 
    farPlane=0.371364, width=0.0222604, height=0.00888881, cameraPosition=(
    0.0825948, -0.152318, -0.277625), cameraUpVector=(-0.220602, 0.894281, 
    -0.389353), cameraTarget=(0.00734809, -0.0418672, 0.0186912), 
    viewOffsetX=-0.0024389, viewOffsetY=0.0397766)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.25553, 
    farPlane=0.392564, width=0.0205569, height=0.00820858, cameraPosition=(
    -0.00107458, 0.128137, -0.301376), cameraUpVector=(0.0683025, 0.853575, 
    0.516473), cameraTarget=(-0.00603004, -0.0398294, -0.0231187), 
    viewOffsetX=-0.00225226, viewOffsetY=0.0367326)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.239673, 
    farPlane=0.408422, width=0.0192813, height=0.0076992, cameraPosition=(
    -0.0335362, 0.17231, -0.276505), cameraUpVector=(0.0547639, 0.767144, 
    0.639133), cameraTarget=(-0.00528772, -0.0361585, -0.0287003), 
    viewOffsetX=-0.0021125, viewOffsetY=0.0344532)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM9', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.28722, 
    farPlane=0.360846, width=0.0231064, height=0.00922657, cameraPosition=(
    -0.292628, -0.0973963, -0.109564), cameraUpVector=(-0.315028, 0.567299, 
    0.760874), cameraTarget=(0.0157354, -0.0282166, -0.0334687), 
    viewOffsetX=-0.00253158, viewOffsetY=0.0412881)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.254086, 
    farPlane=0.393996, width=0.0204408, height=0.00816217, cameraPosition=(
    -0.0977419, 0.125792, 0.286121), cameraUpVector=(0.17741, 0.85953, 
    -0.479306), cameraTarget=(-0.00527733, -0.0403221, 0.0224535), 
    viewOffsetX=-0.00223953, viewOffsetY=0.036525)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.268575, 
    farPlane=0.379505, width=0.0216064, height=0.00862759, cameraPosition=(
    0.0914068, 0.0878302, 0.30192), cameraUpVector=(0.196194, 0.863727, 
    -0.464202), cameraTarget=(-0.00677662, -0.041266, 0.0202135), 
    viewOffsetX=-0.00236723, viewOffsetY=0.0386077)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.268405, 
    farPlane=0.379675, width=0.0215927, height=0.00862213, cameraPosition=(
    0.102763, 0.170474, 0.260089), cameraUpVector=(-0.1624, -0.874366, 
    0.457286), cameraTarget=(0.00457966, 0.0413783, -0.0216175), 
    viewOffsetX=-0.00236573, viewOffsetY=0.0385833)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.306816, 
    farPlane=0.341258, width=0.0246828, height=0.00985604, cameraPosition=(
    -0.0721499, 0.0144085, 0.319017), cameraUpVector=(0.156562, -0.9854, 
    -0.0668984), cameraTarget=(-0.00985842, 0.0458578, 0.00153445), 
    viewOffsetX=-0.00270429, viewOffsetY=0.0441049)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.283536, 
    farPlane=0.36454, width=0.02281, height=0.0091082, cameraPosition=(
    -0.275711, 0.115397, -0.133795), cameraUpVector=(0.166104, -0.750339, 
    -0.639844), cameraTarget=(-0.00536705, 0.0367187, 0.0286493), 
    viewOffsetX=-0.0024991, viewOffsetY=0.0407584)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.304279, 
    farPlane=0.343793, width=0.0244787, height=0.00977454, cameraPosition=(
    -0.130784, 0.0132058, -0.29984), cameraUpVector=(0.00186043, -0.993748, 
    0.111633), cameraTarget=(0.00294364, 0.0465259, -0.00544049), 
    viewOffsetX=-0.00268193, viewOffsetY=0.0437402)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM10', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.306267, 
    farPlane=0.341806, width=0.00860577, height=0.00343635, 
    viewOffsetX=-0.0020659, viewOffsetY=0.0447195)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.305338, 
    farPlane=0.342811, width=0.00857966, height=0.00342593, cameraPosition=(
    -0.0311929, 0.0884962, -0.313738), cameraUpVector=(0.112066, -0.98383, 
    -0.139715), cameraTarget=(-0.00243034, 0.0461839, 0.00727128), 
    viewOffsetX=-0.00205963, viewOffsetY=0.0445838)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.304417, 
    farPlane=0.343732, width=0.0158811, height=0.00634144, 
    viewOffsetX=-0.00180679, viewOffsetY=0.0434478)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.296224, 
    farPlane=0.35174, width=0.0154536, height=0.00617076, cameraPosition=(
    -0.030936, -0.0175891, -0.325346), cameraUpVector=(0.202474, -0.963301, 
    0.176226), cameraTarget=(-0.00665121, 0.0456738, -0.00742629), 
    viewOffsetX=-0.00175816, viewOffsetY=0.0422784)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.225908, 
    farPlane=0.422023, width=0.0117854, height=0.00470598, cameraPosition=(
    0.0654522, -0.212431, -0.240021), cameraUpVector=(-0.0338075, -0.64911, 
    0.759943), cameraTarget=(0.00399935, 0.0316254, -0.034291), 
    viewOffsetX=-0.00134082, viewOffsetY=0.0322426)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.294279, 
    farPlane=0.35427, width=0.0153522, height=0.00613024, cameraPosition=(
    0.0595507, 0.113624, -0.301551), cameraUpVector=(-0.121846, -0.96497, 
    -0.232348), cameraTarget=(0.00810083, 0.044643, 0.0119128), 
    viewOffsetX=-0.00174662, viewOffsetY=0.0420008)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.299503, 
    farPlane=0.348763, width=0.0156247, height=0.00623906, cameraPosition=(
    0.0221036, -0.00932029, -0.326564), cameraUpVector=(0.0303065, -0.984443, 
    0.173072), cameraTarget=(0.00116672, 0.0462215, -0.00696807), 
    viewOffsetX=-0.00177762, viewOffsetY=0.0427464)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.30176, 
    farPlane=0.347465, width=0.0157424, height=0.00628607, cameraPosition=(
    0.213872, 0.0837629, 0.234162), cameraUpVector=(-0.201877, -0.927006, 
    0.316078), cameraTarget=(0.00802352, 0.043895, -0.0142363), 
    viewOffsetX=-0.00179101, viewOffsetY=0.0430685)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.316963, 
    farPlane=0.331278, width=0.0165355, height=0.00660277, cameraPosition=(
    -0.184524, 0.0468429, 0.266434), cameraUpVector=(0.075903, -0.995727, 
    0.052594), cameraTarget=(-0.00447121, 0.0462724, -0.00420542), 
    viewOffsetX=-0.00188124, viewOffsetY=0.0452384)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.264757, 
    farPlane=0.383569, width=0.013812, height=0.00551524, cameraPosition=(
    0.127077, -0.0970148, 0.285712), cameraUpVector=(-0.094483, -0.899456, 
    -0.426674), cameraTarget=(0.00235625, 0.0422351, 0.0197817), 
    viewOffsetX=-0.00157138, viewOffsetY=0.0377873)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.305067, 
    farPlane=0.343396, width=0.0159149, height=0.00635495, cameraPosition=(
    0.0662439, 0.0854248, 0.309262), cameraUpVector=(-0.117289, -0.98237, 
    0.145574), cameraTarget=(0.00347572, 0.0460176, -0.0072379), 
    viewOffsetX=-0.00181063, viewOffsetY=0.0435405)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.30912, 
    farPlane=0.339245, width=0.0161263, height=0.00643938, cameraPosition=(
    -0.0268931, 0.0164892, 0.325982), cameraUpVector=(-0.0542508, -0.993771, 
    -0.0973432), cameraTarget=(0.000749492, 0.0465682, 0.00349735), 
    viewOffsetX=-0.00183468, viewOffsetY=0.044119)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM8', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].view.setValues(width=0.0151573, 
    height=0.00605242, viewOffsetX=-0.00180098, viewOffsetY=0.0441394)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.251391, 
    farPlane=0.39702, width=0.0123279, height=0.00492261, cameraPosition=(
    -0.233384, 0.20583, -0.102634), cameraUpVector=(-0.308665, -0.818258, 
    -0.484953), cameraTarget=(0.0160684, 0.0382716, 0.0213127), 
    viewOffsetX=-0.00146479, viewOffsetY=0.0358998)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.31141, 
    farPlane=0.336968, width=0.0152711, height=0.00609786, cameraPosition=(
    -0.00193433, 0.0677505, -0.320462), cameraUpVector=(-0.399823, -0.915458, 
    -0.0455819), cameraTarget=(0.0203555, 0.0419192, 0.00280385), 
    viewOffsetX=-0.0018145, viewOffsetY=0.0444707)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.259259, 
    farPlane=0.388968, width=0.0127137, height=0.00507666, cameraPosition=(
    -0.171714, -0.110039, -0.256029), cameraUpVector=(0.265305, -0.884345, 
    0.384118), cameraTarget=(-0.0102291, 0.0417068, -0.0182039), 
    viewOffsetX=-0.00151063, viewOffsetY=0.0370233)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.265823, 
    farPlane=0.382567, width=0.0130356, height=0.0052052, cameraPosition=(
    -0.0107101, 0.185715, -0.269706), cameraUpVector=(0.267825, -0.861528, 
    -0.431322), cameraTarget=(-0.0105572, 0.0402304, 0.0209819), 
    viewOffsetX=-0.00154888, viewOffsetY=0.0379607)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.274716, 
    farPlane=0.373667, width=0.0134717, height=0.00537934, cameraPosition=(
    -0.0247038, 0.163118, -0.283046), cameraUpVector=(-0.0997097, -0.927174, 
    -0.361119), cameraTarget=(0.0065922, 0.0427755, 0.0172935), 
    viewOffsetX=-0.0016007, viewOffsetY=0.0392307)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.308222, 
    farPlane=0.340135, width=0.0151148, height=0.00603544, cameraPosition=(
    -0.031477, 0.0129209, -0.325714), cameraUpVector=(0.13748, -0.986356, 
    0.0905635), cameraTarget=(-0.00440639, 0.0462787, -0.00350247), 
    viewOffsetX=-0.00179593, viewOffsetY=0.0440156)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.268791, 
    farPlane=0.379521, width=0.0131812, height=0.00526333, cameraPosition=(
    -0.313902, -0.0773959, -0.0515539), cameraUpVector=(0.402836, -0.889846, 
    -0.214237), cameraTarget=(-0.0176627, 0.0423114, 0.00826408), 
    viewOffsetX=-0.00156618, viewOffsetY=0.0383847)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.246491, 
    farPlane=0.401781, width=0.0120876, height=0.00482665, cameraPosition=(
    -0.00425685, -0.158284, 0.28646), cameraUpVector=(0.185923, -0.783727, 
    -0.592625), cameraTarget=(-0.0104809, 0.0367982, 0.0265178), 
    viewOffsetX=-0.00143624, viewOffsetY=0.0352001)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.281835, 
    farPlane=0.366451, width=0.0138208, height=0.00551874, cameraPosition=(
    -0.0976786, -0.0540978, 0.307766), cameraUpVector=(0.143232, -0.951294, 
    -0.27299), cameraTarget=(-0.00817191, 0.0444769, 0.0112238), 
    viewOffsetX=-0.00164218, viewOffsetY=0.0402474)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.281584, 
    farPlane=0.366702, width=0.0138085, height=0.00551384, cameraPosition=(
    -0.0905687, -0.141852, 0.280741), cameraUpVector=(0.0675502, 0.940496, 
    0.333024), cameraTarget=(-0.00106204, -0.0432773, -0.0158009), 
    viewOffsetX=-0.00164072, viewOffsetY=0.0402116)
session.viewports['Viewport: 1'].view.setValues(session.views['Back'])
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.498238, 
    farPlane=0.505683, width=0.0234639, height=0.00936931, 
    viewOffsetX=-0.000967492, viewOffsetY=0.000459559)
session.viewports['Viewport: 1'].odbDisplay.setValues(viewCut=ON)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].viewCutOptions.setValues(
    summationPoint=(0.0, 0.003, 0.0))
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].viewCutOptions.setValues(
    summationPoint=(0.0, 0.0053, 0.0))
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    angle=78)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.497343, 
    farPlane=0.506578, width=0.0265072, height=0.0105845, 
    viewOffsetX=-0.0011492, viewOffsetY=0.000416272)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    origin=(0, 0.004, 0))
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.493763, 
    farPlane=0.510158, width=0.0496925, height=0.0198426, 
    viewOffsetX=-0.00128028, viewOffsetY=0.00125664)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    angle=74)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    origin=(0, 0.005, 0))
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    angle=75)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.493182, 
    farPlane=0.510739, width=0.0534528, height=0.0213441, 
    viewOffsetX=-0.000389684, viewOffsetY=-0.00147519)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.45657, 
    farPlane=0.545943, width=0.0494847, height=0.0197596, cameraPosition=(
    -0.4334, 0.120815, -0.221095), cameraUpVector=(0.215644, 0.970701, 
    0.106004), cameraTarget=(0.000546191, 0.00021168, 0.000512543), 
    viewOffsetX=-0.000360755, viewOffsetY=-0.00136568)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.454079, 
    farPlane=0.548141, width=0.0492148, height=0.0196518, cameraPosition=(
    -0.483872, 0.130141, 0.0101688), cameraUpVector=(0.256149, 0.936456, 
    0.239663), cameraTarget=(0.000840542, 5.19895e-05, 0.00042003), 
    viewOffsetX=-0.000358787, viewOffsetY=-0.00135823)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.459424, 
    farPlane=0.542797, width=0.0119983, height=0.004791, 
    viewOffsetX=-0.00106679, viewOffsetY=-0.000162631)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.460799, 
    farPlane=0.541849, width=0.0120342, height=0.00480534, cameraPosition=(
    -0.465788, 0.126779, -0.135476), cameraUpVector=(0.23117, 0.967048, 
    0.106669), cameraTarget=(0.00051967, 0.000296229, 0.000627136), 
    viewOffsetX=-0.00106998, viewOffsetY=-0.000163118)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.460794, 
    farPlane=0.542004, width=0.0120341, height=0.00480529, cameraPosition=(
    -0.448119, 0.128005, -0.185108), cameraUpVector=(0.237221, 0.967102, 
    0.0918679), cameraTarget=(0.000390868, 0.000343767, 0.000650083), 
    viewOffsetX=-0.00106997, viewOffsetY=-0.000163116)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    angle=80)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.458631, 
    farPlane=0.544167, width=0.0285992, height=0.0114199, 
    viewOffsetX=-0.0014776, viewOffsetY=0.000117944)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.488861, 
    farPlane=0.515892, width=0.0304842, height=0.0121726, cameraPosition=(
    -0.0693552, 0.0317376, -0.496569), cameraUpVector=(0.00989486, 0.998053, 
    0.0615744), cameraTarget=(-0.000989638, 0.000440667, -0.000271251), 
    viewOffsetX=-0.001575, viewOffsetY=0.000125718)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.453178, 
    farPlane=0.550851, width=0.0282591, height=0.0112841, cameraPosition=(
    -0.372941, 0.154756, -0.29843), cameraUpVector=(0.298291, 0.947331, 
    0.116558), cameraTarget=(-0.0006475, 0.000694614, 0.000949041), 
    viewOffsetX=-0.00146004, viewOffsetY=0.000116542)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.473481, 
    farPlane=0.530101, width=0.0295252, height=0.0117897, cameraPosition=(
    -0.470038, 0.0682772, -0.16195), cameraUpVector=(0.132272, 0.99075, 
    0.0303224), cameraTarget=(-0.000192216, 0.000555576, 0.0012189), 
    viewOffsetX=-0.00152545, viewOffsetY=0.000121763)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.476766, 
    farPlane=0.526817, width=0.00605153, height=0.00241642, 
    viewOffsetX=-0.00141602, viewOffsetY=0.000866857)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.45537, 
    farPlane=0.54928, width=0.00577995, height=0.00230798, cameraPosition=(
    -0.154739, 0.173819, -0.445265), cameraUpVector=(-0.0311346, 0.927649, 
    0.372153), cameraTarget=(-0.00108235, 0.000344555, 1.19413e-06), 
    viewOffsetX=-0.00135248, viewOffsetY=0.000827954)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.471479, 
    farPlane=0.533267, width=0.00598442, height=0.00238963, cameraPosition=(
    0.0742457, 0.114127, -0.483633), cameraUpVector=(-0.207537, 0.958831, 
    0.193835), cameraTarget=(-0.000778586, 0.000229388, -0.000554968), 
    viewOffsetX=-0.00140032, viewOffsetY=0.000857243)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.469474, 
    farPlane=0.535273, width=0.0207938, height=0.00830312, 
    viewOffsetX=-0.00217306, viewOffsetY=0.000111604)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.457768, 
    farPlane=0.546933, width=0.0202754, height=0.0080961, cameraPosition=(
    0.0499861, 0.161958, -0.472981), cameraUpVector=(-0.133479, 0.942095, 
    0.307637), cameraTarget=(-0.000851706, 0.000447717, -0.000438921), 
    viewOffsetX=-0.00211888, viewOffsetY=0.000108822)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.469844, 
    farPlane=0.53445, width=0.0208103, height=0.00830967, cameraPosition=(
    0.0580926, -0.111935, -0.485996), cameraUpVector=(-0.117594, 0.964424, 
    -0.236765), cameraTarget=(-0.000847876, 0.000123954, -0.000264656), 
    viewOffsetX=-0.00217478, viewOffsetY=0.000111693)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.473342, 
    farPlane=0.528392, width=0.0209652, height=0.00837158, cameraPosition=(
    -0.464254, 0.0694176, -0.174788), cameraUpVector=(0.08384, 0.982865, 
    0.164154), cameraTarget=(0.000369329, 0.000233164, 0.00214486), 
    viewOffsetX=-0.00219097, viewOffsetY=0.000112525)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.473674, 
    farPlane=0.528061, width=0.0174256, height=0.00695817, 
    viewOffsetX=-0.00187636, viewOffsetY=0.000243628)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.475685, 
    farPlane=0.526166, width=0.0174996, height=0.00698772, cameraPosition=(
    -0.26905, 0.0728113, -0.416262), cameraUpVector=(-0.197823, 0.936253, 
    0.290337), cameraTarget=(-0.000885216, -0.000154842, 0.00174582), 
    viewOffsetX=-0.00188433, viewOffsetY=0.000244663)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.435149, 
    farPlane=0.567194, width=0.0160083, height=0.00639225, cameraPosition=(
    -0.140199, 0.244062, -0.414821), cameraUpVector=(-0.0248663, 0.858458, 
    0.51228), cameraTarget=(-0.00147287, -0.000175969, 0.00119585), 
    viewOffsetX=-0.00172375, viewOffsetY=0.000223814)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.433867, 
    farPlane=0.568476, width=0.0279697, height=0.0111685, 
    viewOffsetX=-0.00167245, viewOffsetY=-0.00135471)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.433625, 
    farPlane=0.568718, width=0.0279541, height=0.0111623, cameraPosition=(
    -0.139332, 0.244818, -0.414666), cameraUpVector=(0.414091, 0.838499, 
    0.354187), cameraTarget=(-0.000605402, 0.00058002, 0.00135041), 
    viewOffsetX=-0.00167151, viewOffsetY=-0.00135396)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.470045, 
    farPlane=0.533145, width=0.030302, height=0.0120998, cameraPosition=(
    -0.0731814, 0.104893, -0.485071), cameraUpVector=(0.404525, 0.904811, 
    0.132956), cameraTarget=(-0.000794182, 0.00115067, 0.000690857), 
    viewOffsetX=-0.0018119, viewOffsetY=-0.00146768)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.488903, 
    farPlane=0.51503, width=0.0315177, height=0.0125852, cameraPosition=(
    0.0227799, -0.0306459, -0.5005), cameraUpVector=(-0.0877431, 0.993943, 
    -0.0661661), cameraTarget=(-0.00181388, 0.000493938, -0.000107854), 
    viewOffsetX=-0.00188459, viewOffsetY=-0.00152656)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.487382, 
    farPlane=0.516223, width=0.0314196, height=0.012546, cameraPosition=(
    0.00445561, 0.0376668, -0.500389), cameraUpVector=(-0.000241178, 0.997281, 
    0.0736944), cameraTarget=(-0.00168651, 0.000676443, 0.000171406), 
    viewOffsetX=-0.00187873, viewOffsetY=-0.00152181)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.470658, 
    farPlane=0.532444, width=0.0303415, height=0.0121155, cameraPosition=(
    -0.246833, 0.0901309, -0.427257), cameraUpVector=(0.527225, 0.839766, 
    -0.129715), cameraTarget=(-0.000186095, 0.00140486, 0.000831609), 
    viewOffsetX=-0.00181426, viewOffsetY=-0.00146959)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.479461, 
    farPlane=0.523874, width=0.030909, height=0.0123421, cameraPosition=(
    -0.448301, -0.0424496, -0.221067), cameraUpVector=(-0.092163, 0.995715, 
    -0.00760591), cameraTarget=(-0.000505954, 0.000698874, 0.00160896), 
    viewOffsetX=-0.00184819, viewOffsetY=-0.00149708)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.471781, 
    farPlane=0.531457, width=0.0304139, height=0.0121444, cameraPosition=(
    -0.494975, -0.0706718, -0.0398527), cameraUpVector=(-0.148593, 0.985659, 
    0.0799767), cameraTarget=(0.000152635, 0.000589194, 0.0018275), 
    viewOffsetX=-0.00181858, viewOffsetY=-0.0014731)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.486924, 
    farPlane=0.516398, width=0.0313901, height=0.0125343, cameraPosition=(
    -0.307749, 0.0213687, -0.395612), cameraUpVector=(-0.130167, 0.979706, 
    0.15242), cameraTarget=(-0.00132222, 0.000314009, 0.00140967), 
    viewOffsetX=-0.00187695, viewOffsetY=-0.00152038)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.451649, 
    farPlane=0.551352, width=0.0291161, height=0.0116263, cameraPosition=(
    0.0506879, 0.176791, -0.466661), cameraUpVector=(-0.0953999, 0.934749, 
    0.342263), cameraTarget=(-0.00188719, 0.000426855, 0.00034904), 
    viewOffsetX=-0.00174098, viewOffsetY=-0.00141024)
session.viewports['Viewport: 1'].view.setValues(session.views['Front'])
session.viewports['Viewport: 1'].view.setValues(session.views['Back'])
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.340068, 
    farPlane=0.353443, width=0.0399366, height=0.015947, 
    viewOffsetX=1.32165e-05, viewOffsetY=0.0368578)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM1', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM5', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.284957, 
    farPlane=0.407447, width=0.0334646, height=0.0133627, cameraPosition=(
    -0.203288, 0.126746, -0.252936), cameraUpVector=(0.857681, 0.375768, 
    -0.350973), cameraTarget=(-0.0321994, -0.0151989, 0.0131833), 
    viewOffsetX=1.10747e-05, viewOffsetY=0.0308848)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.306753, 
    farPlane=0.38583, width=0.0360242, height=0.0143848, cameraPosition=(
    -0.335295, 0.0940775, -0.0052442), cameraUpVector=(-0.0245904, -0.525371, 
    -0.850518), cameraTarget=(0.00116451, 0.0187207, 0.031576), 
    viewOffsetX=1.19218e-05, viewOffsetY=0.0332471)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.306381, 
    farPlane=0.386202, width=0.0359805, height=0.0143673, cameraPosition=(
    -0.327495, 0.110383, -0.0431523), cameraUpVector=(-0.232356, -0.959437, 
    0.159659), cameraTarget=(0.0089648, 0.0350257, -0.00633213), 
    viewOffsetX=1.19073e-05, viewOffsetY=0.0332067)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.334309, 
    farPlane=0.359011, width=0.0392603, height=0.0156769, cameraPosition=(
    -0.302102, 0.0324149, -0.17092), cameraUpVector=(-0.211001, -0.881966, 
    0.421444), cameraTarget=(0.00812376, 0.0322226, -0.0160019), 
    viewOffsetX=1.29927e-05, viewOffsetY=0.0362336)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.293848, 
    farPlane=0.400668, width=0.0345086, height=0.0137795, cameraPosition=(
    -0.0737475, -0.0941687, -0.327992), cameraUpVector=(-0.282931, -0.860702, 
    0.423251), cameraTarget=(0.0107566, 0.031457, -0.0160355), 
    viewOffsetX=1.14202e-05, viewOffsetY=0.0318483)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.322889, 
    farPlane=0.371186, width=0.0379191, height=0.0151413, cameraPosition=(
    -0.336609, 0.00413516, -0.091913), cameraUpVector=(0.0595593, -0.988634, 
    0.138042), cameraTarget=(-0.00248617, 0.0364039, -0.00496378), 
    viewOffsetX=1.25489e-05, viewOffsetY=0.0349959)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.32848, 
    farPlane=0.365637, width=0.0385757, height=0.0154035, cameraPosition=(
    -0.348315, 0.0217216, -0.000532902), cameraUpVector=(0.043479, -0.998902, 
    0.0174278), cameraTarget=(-0.00188784, 0.0368027, -0.000356037), 
    viewOffsetX=1.27662e-05, viewOffsetY=0.0356019)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.330669, 
    farPlane=0.363446, width=0.0243175, height=0.00971018, 
    viewOffsetX=-8.32832e-05, viewOffsetY=0.0348306)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.334672, 
    farPlane=0.359479, width=0.0246119, height=0.00982771, cameraPosition=(
    -0.347059, 0.0334354, 0.015696), cameraUpVector=(0.00912227, -0.999811, 
    -0.0171694), cameraTarget=(-0.000635709, 0.0368491, 0.000921848), 
    viewOffsetX=-8.42912e-05, viewOffsetY=0.0352522)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.331535, 
    farPlane=0.362856, width=0.0243812, height=0.0097356, cameraPosition=(
    -0.314183, 0.0147693, 0.151575), cameraUpVector=(-0.214156, -0.759618, 
    -0.6141), cameraTarget=(0.00752688, 0.0281141, 0.0228763), 
    viewOffsetX=-8.35011e-05, viewOffsetY=0.0349218)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.279833, 
    farPlane=0.414242, width=0.020579, height=0.00821736, cameraPosition=(
    -0.289535, 0.187161, -0.0549735), cameraUpVector=(-0.280512, -0.791287, 
    -0.543302), cameraTarget=(0.0100451, 0.0294688, 0.0200181), 
    viewOffsetX=-7.04793e-05, viewOffsetY=0.0294758)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.302751, 
    farPlane=0.391826, width=0.0222644, height=0.00889035, cameraPosition=(
    -0.188484, 0.113869, 0.271151), cameraUpVector=(-0.775133, -0.342781, 
    -0.530725), cameraTarget=(0.0281487, 0.0130698, 0.0198569), 
    viewOffsetX=-7.62515e-05, viewOffsetY=0.0318898)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.302506, 
    farPlane=0.392071, width=0.0222464, height=0.00888315, cameraPosition=(
    -0.191301, 0.0936283, 0.276841), cameraUpVector=(-0.697047, 0.210627, 
    -0.685392), cameraTarget=(0.0253312, -0.00717094, 0.025547), 
    viewOffsetX=-7.61898e-05, viewOffsetY=0.031864)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.313343, 
    farPlane=0.381027, width=0.0230433, height=0.00920137, cameraPosition=(
    -0.331845, 0.0625745, 0.0887404), cameraUpVector=(-0.143763, 0.0528584, 
    -0.988199), cameraTarget=(0.00489242, -0.00145201, 0.0363262), 
    viewOffsetX=-7.89192e-05, viewOffsetY=0.0330055)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.312413, 
    farPlane=0.381899, width=0.0229749, height=0.00917406, cameraPosition=(
    -0.343117, 0.064338, 0.00478066), cameraUpVector=(0.104815, 0.0628197, 
    -0.992506), cameraTarget=(-0.00420143, -0.00181381, 0.0363844), 
    viewOffsetX=-7.8685e-05, viewOffsetY=0.0329075)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.308114, 
    farPlane=0.386183, width=0.0226587, height=0.00904782, cameraPosition=(
    -0.339153, 0.0792683, -0.0242102), cameraUpVector=(0.175692, -0.0117929, 
    -0.984374), cameraTarget=(-0.00678205, 0.000928181, 0.0360493), 
    viewOffsetX=-7.76022e-05, viewOffsetY=0.0324547)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.323162, 
    farPlane=0.371393, width=0.0237654, height=0.00948972, cameraPosition=(
    -0.00166046, 0.0510403, 0.345481), cameraUpVector=(-0.987967, 0.0922113, 
    -0.124174), cameraTarget=(0.0360933, -0.00290176, 0.00503496), 
    viewOffsetX=-8.13924e-05, viewOffsetY=0.0340398)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.308813, 
    farPlane=0.385763, width=0.0227102, height=0.00906836, cameraPosition=(
    -0.240806, 0.0844023, 0.238486), cameraUpVector=(-0.6393, -0.105464, 
    -0.761691), cameraTarget=(0.0229982, 0.00436716, 0.0281516), 
    viewOffsetX=-7.77784e-05, viewOffsetY=0.0325284)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.308003, 
    farPlane=0.386576, width=0.0226506, height=0.00904457, cameraPosition=(
    -0.266117, 0.0832825, 0.210315), cameraUpVector=(-0.538595, -0.0342907, 
    -0.841866), cameraTarget=(0.0192871, 0.0017662, 0.0310431), 
    viewOffsetX=-7.75743e-05, viewOffsetY=0.0324431)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.316956, 
    farPlane=0.377623, width=0.023309, height=0.00930746, cameraPosition=(
    -0.318477, 0.0514802, 0.133759), cameraUpVector=(-0.272024, 0.0953384, 
    -0.957556), cameraTarget=(0.00946974, -0.00301646, 0.0351684), 
    viewOffsetX=-7.98291e-05, viewOffsetY=0.0333861)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.318291, 
    farPlane=0.376288, width=0.0143267, height=0.00572074, 
    viewOffsetX=-0.00191844, viewOffsetY=0.0339263)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.318167, 
    farPlane=0.376412, width=0.0143211, height=0.00571852, cameraPosition=(
    -0.31772, 0.0567766, 0.133351), cameraUpVector=(-0.29481, -0.0475443, 
    -0.954372), cameraTarget=(0.0102271, 0.00227993, 0.0347602), 
    viewOffsetX=-0.00191769, viewOffsetY=0.0339132)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.334953, 
    farPlane=0.360591, width=0.0150767, height=0.00602022, cameraPosition=(
    -0.3495, 0.0107734, -0.00312302), cameraUpVector=(0.111814, -0.0457221, 
    -0.992677), cameraTarget=(-0.00507369, 0.00216739, 0.0360678), 
    viewOffsetX=-0.00201886, viewOffsetY=0.0357024)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.334843, 
    farPlane=0.360701, width=0.0150718, height=0.00601825, cameraPosition=(
    -0.34957, 0.00864892, -0.00297343), cameraUpVector=(0.11334, 0.0116452, 
    -0.993488), cameraTarget=(-0.00514379, 4.29061e-05, 0.0362174), 
    viewOffsetX=-0.0020182, viewOffsetY=0.0356907)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.334844, 
    farPlane=0.3607, width=0.0150718, height=0.00601827, cameraPosition=(
    -0.3496, 0.00757945, -0.00294408), cameraUpVector=(0.113966, 0.0404537, 
    -0.992661), cameraTarget=(-0.00517385, -0.00102657, 0.0362467), 
    viewOffsetX=-0.00201821, viewOffsetY=0.0356908)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.326039, 
    farPlane=0.369243, width=0.0146755, height=0.00586002, cameraPosition=(
    -0.348049, 0.0324446, -0.00378951), cameraUpVector=(0.117928, 0.0215863, 
    -0.992788), cameraTarget=(-0.00516717, -0.000264009, 0.0362269), 
    viewOffsetX=-0.00196514, viewOffsetY=0.0347523)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.335474, 
    farPlane=0.360287, width=0.0151002, height=0.00602962, cameraPosition=(
    -0.349198, -0.0091046, -0.0182914), cameraUpVector=(0.15509, 0.0583386, 
    -0.986176), cameraTarget=(-0.00679712, -0.00172245, 0.0359912), 
    viewOffsetX=-0.00202201, viewOffsetY=0.035758)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.329283, 
    farPlane=0.366087, width=0.0148215, height=0.00591834, cameraPosition=(
    -0.348824, 0.0235359, -0.000174008), cameraUpVector=(0.106833, 0.0213083, 
    -0.994049), cameraTarget=(-0.00481153, -0.000249626, 0.0362867), 
    viewOffsetX=-0.00198469, viewOffsetY=0.0350981)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.329323, 
    farPlane=0.366047, width=0.0148233, height=0.00591907, cameraPosition=(
    -0.348753, 0.024503, -0.000213501), cameraUpVector=(0.105072, -0.00477622, 
    -0.994453), cameraTarget=(-0.00474048, 0.000717507, 0.0362472), 
    viewOffsetX=-0.00198493, viewOffsetY=0.0351024)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.330676, 
    farPlane=0.364694, width=0.0058791, height=0.00234757, 
    viewOffsetX=-0.00127928, viewOffsetY=0.0348652)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.332134, 
    farPlane=0.363263, width=0.00590502, height=0.00235792, cameraPosition=(
    -0.349034, 0.0202972, -3.7059e-05), cameraUpVector=(0.104607, -0.00371434, 
    -0.994507), cameraTarget=(-0.00474036, 0.00066846, 0.0362493), 
    viewOffsetX=-0.00128492, viewOffsetY=0.0350189)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.331667, 
    farPlane=0.363731, width=0.00854758, height=0.00341312, 
    viewOffsetX=-0.000870602, viewOffsetY=0.0351245)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM1', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM5', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM1', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM8', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.329181, 
    farPlane=0.366218, width=0.027827, height=0.0111115, 
    viewOffsetX=-0.00304561, viewOffsetY=0.0348661)
o3 = session.odbs['C:/LocalUserData/User-data/nguyenb5/CP1000 plastic (UMAT UMATHT)/CP1000_dense_CSC/CP1000_CHD4_dense_CSC/CHD4_combined_processed.odb']
session.viewports['Viewport: 1'].setValues(displayedObject=o3)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.070792, 
    farPlane=0.327935, width=0.0231394, height=0.00923976, 
    viewOffsetX=-0.00142518, viewOffsetY=0.000175433)
session.viewports['Viewport: 1'].view.setValues(session.views['Front'])
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.489014, 
    farPlane=0.512792, width=0.0736713, height=0.0294175, 
    viewOffsetX=0.00499903, viewOffsetY=-0.000322151)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM1', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].odbDisplay.display.setValues(
    plotState=CONTOURS_ON_DEF)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.49897, 
    farPlane=0.502836, width=0.0104638, height=0.00417827, 
    viewOffsetX=-0.000302142, viewOffsetY=0.000307158)
session.viewports['Viewport: 1'].odbDisplay.setValues(viewCut=ON)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    origin=(-0.002, 0.000, 0))
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    angle=0)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    motion=TRANSLATE)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    position=0)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.49791, 
    farPlane=0.503896, width=0.0161017, height=0.00642951, 
    viewOffsetX=-0.000618233, viewOffsetY=-0.000366484)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.472027, 
    farPlane=0.529348, width=0.0152646, height=0.00609529, cameraPosition=(
    -0.0111542, 0.102198, 0.490177), cameraUpVector=(-0.000623838, 0.97895, 
    -0.204102), cameraTarget=(-3.84054e-07, -4.55469e-06, -6.17713e-05), 
    viewOffsetX=-0.000586096, viewOffsetY=-0.000347434)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.471698, 
    farPlane=0.529678, width=0.0195376, height=0.00780153, 
    viewOffsetX=-0.000627827, viewOffsetY=-0.000175411)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    position=0.0108154)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.472928, 
    farPlane=0.528447, width=0.00932262, height=0.00372259, 
    viewOffsetX=-0.00160909, viewOffsetY=0.00056276)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    position=0.0032701)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    position=0.0032701)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.470227, 
    farPlane=0.531148, width=0.0267561, height=0.0106839, 
    viewOffsetX=-0.00224169, viewOffsetY=0.00133105)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    position=0.000754997)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.473973, 
    farPlane=0.527402, width=0.00290718, height=0.00116086, 
    viewOffsetX=-0.00214044, viewOffsetY=0.000558464)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.382652, 
    farPlane=0.619115, width=0.00234705, height=0.000937196, cameraPosition=(
    -0.0754661, 0.46456, 0.17342), cameraUpVector=(0.0128615, 0.350797, 
    -0.936363), cameraTarget=(-6.11914e-06, 0.000507677, 0.000604153), 
    viewOffsetX=-0.00172804, viewOffsetY=0.000450864)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.38111, 
    farPlane=0.620657, width=0.0143608, height=0.00573436, 
    viewOffsetX=-0.00269622, viewOffsetY=0.00109594)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.375096, 
    farPlane=0.62785, width=0.0141341, height=0.00564387, cameraPosition=(
    -0.11069, 0.489756, -0.010071), cameraUpVector=(-0.0640307, -0.0355355, 
    -0.997315), cameraTarget=(5.1609e-05, 0.00135571, 0.000220549), 
    viewOffsetX=-0.00265367, viewOffsetY=0.00107865)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.380516, 
    farPlane=0.620108, width=0.0143383, height=0.00572543, cameraPosition=(
    0.0132268, 0.467416, 0.179903), cameraUpVector=(-0.0272049, 0.358866, 
    -0.932993), cameraTarget=(0.000231893, -6.56921e-05, 0.000469699), 
    viewOffsetX=-0.00269202, viewOffsetY=0.00109424)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.439995, 
    farPlane=0.559845, width=0.0165795, height=0.00662038, cameraPosition=(
    0.00759127, 0.227559, 0.445447), cameraUpVector=(0.0256933, 0.889744, 
    -0.455736), cameraTarget=(0.000163304, -0.000600572, -0.000414022), 
    viewOffsetX=-0.00311281, viewOffsetY=0.00126528)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.459382, 
    farPlane=0.540845, width=0.01731, height=0.00691209, cameraPosition=(
    -0.0348914, 0.147942, 0.476691), cameraUpVector=(0.100398, 0.951977, 
    -0.289242), cameraTarget=(0.000114194, -0.000695852, -0.000368674), 
    viewOffsetX=-0.00324997, viewOffsetY=0.00132103)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.459276, 
    farPlane=0.540952, width=0.017306, height=0.0069105, cameraPosition=(
    -0.034684, 0.148391, 0.476566), cameraUpVector=(-0.0326496, 0.95354, 
    -0.299492), cameraTarget=(0.000321568, -0.000247235, -0.000493234), 
    viewOffsetX=-0.00324922, viewOffsetY=0.00132073)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    motion=ROTATE)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.458765, 
    farPlane=0.541463, width=0.0221415, height=0.00884127, 
    viewOffsetX=-0.00305383, viewOffsetY=0.00146129)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    angle=180)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    origin=(-0.002, 0.002, 0))
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    origin=(-0.002, 0.001, 0))
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    origin=(-0.002, 0.0008, 0))
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.460151, 
    farPlane=0.540076, width=0.0120107, height=0.00479598, 
    viewOffsetX=-0.00263919, viewOffsetY=0.00125859)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    origin=(-0.002, 0.0007, 0))
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.459692, 
    farPlane=0.540536, width=0.0153682, height=0.00613665, 
    viewOffsetX=-0.00265328, viewOffsetY=0.00142185)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.422554, 
    farPlane=0.577816, width=0.0141266, height=0.00564088, cameraPosition=(
    -0.127855, -0.291391, 0.385346), cameraUpVector=(0.179591, 0.756752, 
    0.628548), cameraTarget=(4.02392e-06, -0.00026354, -0.001695), 
    viewOffsetX=-0.00243893, viewOffsetY=0.00130698)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.378768, 
    farPlane=0.622008, width=0.0126628, height=0.00505635, cameraPosition=(
    -0.123989, -0.466366, -0.12969), cameraUpVector=(0.18607, -0.303256, 
    0.934566), cameraTarget=(-9.24441e-06, 0.00196293, -0.00240644), 
    viewOffsetX=-0.0021862, viewOffsetY=0.00117155)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.372102, 
    farPlane=0.626447, width=0.0124399, height=0.00496736, cameraPosition=(
    0.0585153, -0.493235, 0.0427382), cameraUpVector=(-0.19114, 0.0648572, 
    0.979418), cameraTarget=(0.000456201, 0.00232926, -0.0014085), 
    viewOffsetX=-0.00214772, viewOffsetY=0.00115093)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.410047, 
    farPlane=0.590764, width=0.0137084, height=0.0054739, cameraPosition=(
    -0.148967, -0.339982, -0.334838), cameraUpVector=(0.185442, -0.726042, 
    0.662174), cameraTarget=(0.000438446, 0.00246393, -0.00120313), 
    viewOffsetX=-0.00236673, viewOffsetY=0.00126829)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.435671, 
    farPlane=0.563466, width=0.014565, height=0.00581597, cameraPosition=(
    0.0465387, -0.238944, -0.435834), cameraUpVector=(0.0838321, -0.867471, 
    0.490374), cameraTarget=(0.000563407, 0.00313773, 0.000269066), 
    viewOffsetX=-0.00251463, viewOffsetY=0.00134755)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.477364, 
    farPlane=0.522243, width=0.0159589, height=0.00637255, cameraPosition=(
    0.000741195, -0.0762971, -0.493832), cameraUpVector=(0.0422831, -0.986613, 
    0.157502), cameraTarget=(0.000769896, 0.0026676, 0.000807142), 
    viewOffsetX=-0.00275528, viewOffsetY=0.00147651)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.474175, 
    farPlane=0.524724, width=0.0158523, height=0.00632997, cameraPosition=(
    0.087081, -0.081775, -0.484828), cameraUpVector=(-0.033492, -0.985721, 
    0.165022), cameraTarget=(0.00065503, 0.00254592, 0.00130314), 
    viewOffsetX=-0.00273687, viewOffsetY=0.00146664)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    angle=161)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.474458, 
    farPlane=0.52444, width=0.0131746, height=0.00526071, 
    viewOffsetX=-0.00323833, viewOffsetY=0.00142027)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.410856, 
    farPlane=0.588945, width=0.0114085, height=0.00455551, cameraPosition=(
    -0.0343488, -0.341467, -0.362776), cameraUpVector=(-0.282168, -0.682746, 
    0.673972), cameraTarget=(0.00128649, 0.00199828, 7.95597e-05), 
    viewOffsetX=-0.00280423, viewOffsetY=0.00122988)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.37684, 
    farPlane=0.622853, width=0.0104639, height=0.00417834, cameraPosition=(
    -0.0578429, -0.477101, -0.134754), cameraUpVector=(0.0694436, -0.274853, 
    0.958975), cameraTarget=(0.000922353, 0.0021995, -0.00163728), 
    viewOffsetX=-0.00257206, viewOffsetY=0.00112805)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.386737, 
    farPlane=0.611601, width=0.0107387, height=0.00428807, cameraPosition=(
    0.0902362, -0.431882, -0.232076), cameraUpVector=(0.270838, -0.406068, 
    0.872786), cameraTarget=(0.000154001, 0.00358516, -0.00152051), 
    viewOffsetX=-0.00263961, viewOffsetY=0.00115768)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.37267, 
    farPlane=0.62683, width=0.0103481, height=0.0041321, cameraPosition=(
    -0.0678072, -0.493905, -0.0216715), cameraUpVector=(0.106008, -0.0536121, 
    0.992919), cameraTarget=(0.00108346, 0.00185631, -0.00225948), 
    viewOffsetX=-0.0025436, viewOffsetY=0.00111557)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.395885, 
    farPlane=0.604573, width=0.0109927, height=0.0043895, cameraPosition=(
    -0.110712, -0.398738, -0.279978), cameraUpVector=(0.404236, -0.592858, 
    0.6965), cameraTarget=(0.000401927, 0.00313611, -0.00239447), 
    viewOffsetX=-0.00270205, viewOffsetY=0.00118506)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.394594, 
    farPlane=0.605863, width=0.020426, height=0.00815626, 
    viewOffsetX=-0.00321317, viewOffsetY=0.00204192)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.476225, 
    farPlane=0.522668, width=0.0246516, height=0.00984357, cameraPosition=(
    -0.0206453, -0.0736802, -0.493447), cameraUpVector=(0.0663161, -0.986257, 
    0.151328), cameraTarget=(0.00160321, 0.00367481, 0.000944438), 
    viewOffsetX=-0.00387789, viewOffsetY=0.00246434)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.472474, 
    farPlane=0.525958, width=0.0244575, height=0.00976605, cameraPosition=(
    0.0290404, -0.0849309, -0.490959), cameraUpVector=(0.047545, -0.982597, 
    0.179562), cameraTarget=(0.00150941, 0.00368935, 0.00127044), 
    viewOffsetX=-0.00384735, viewOffsetY=0.00244493)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    angle=152)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.384593, 
    farPlane=0.61479, width=0.0199084, height=0.00794957, cameraPosition=(
    -0.0648281, -0.436581, -0.232886), cameraUpVector=(-0.0186539, -0.463191, 
    0.886062), cameraTarget=(0.00197141, 0.00278183, -0.00180292), 
    viewOffsetX=-0.00313174, viewOffsetY=0.00199017)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.370708, 
    farPlane=0.62825, width=0.0191896, height=0.00766256, cameraPosition=(
    -0.0136005, -0.498378, 0.0130812), cameraUpVector=(0.150022, 0.0274302, 
    0.988302), cameraTarget=(0.0013994, 0.00203625, -0.00308576), 
    viewOffsetX=-0.00301867, viewOffsetY=0.00191832)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.380042, 
    farPlane=0.620381, width=0.0196728, height=0.00785549, cameraPosition=(
    -0.179384, -0.454796, -0.102556), cameraUpVector=(0.143632, -0.265033, 
    0.953482), cameraTarget=(0.00161462, 0.00152196, -0.00298289), 
    viewOffsetX=-0.00309467, viewOffsetY=0.00196662)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.464102, 
    farPlane=0.536452, width=0.0240242, height=0.00959302, cameraPosition=(
    -0.122414, -0.119852, -0.46985), cameraUpVector=(-0.0916968, -0.956787, 
    0.275952), cameraTarget=(0.00232087, 0.00351181, -0.000676096), 
    viewOffsetX=-0.00377917, viewOffsetY=0.00240161)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.483142, 
    farPlane=0.517278, width=0.0250098, height=0.00998657, cameraPosition=(
    -0.0447774, 0.0529712, -0.495472), cameraUpVector=(-0.0865151, -0.99215, 
    -0.0902928), cameraTarget=(0.00232344, 0.00370363, 0.000769637), 
    viewOffsetX=-0.00393421, viewOffsetY=0.00250014)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.417756, 
    farPlane=0.582956, width=0.0216251, height=0.00863504, cameraPosition=(
    -0.194759, -0.299879, -0.34936), cameraUpVector=(-0.326007, -0.615398, 
    0.717639), cameraTarget=(0.00261102, 0.00145195, -0.00130021), 
    viewOffsetX=-0.00340178, viewOffsetY=0.00216179)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.394378, 
    farPlane=0.605985, width=0.020415, height=0.00815182, cameraPosition=(
    -0.173515, -0.395574, -0.251013), cameraUpVector=(-0.12104, -0.48822, 
    0.864286), cameraTarget=(0.0023387, 0.00182056, -0.00190558), 
    viewOffsetX=-0.00321142, viewOffsetY=0.00204082)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.459225, 
    farPlane=0.541645, width=0.0237718, height=0.00949222, cameraPosition=(
    -0.17378, -0.135782, -0.44901), cameraUpVector=(0.127636, -0.959758, 
    0.250147), cameraTarget=(0.00169207, 0.00426041, -0.0012385), 
    viewOffsetX=-0.00373947, viewOffsetY=0.00237639)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.485329, 
    farPlane=0.514863, width=0.0251231, height=0.0100318, cameraPosition=(
    -0.0566634, -0.0354743, -0.495568), cameraUpVector=(-0.0109645, -0.996671, 
    0.0807863), cameraTarget=(0.00220789, 0.00407175, 0.000286496), 
    viewOffsetX=-0.00395203, viewOffsetY=0.00251147)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.48427, 
    farPlane=0.515595, width=0.0250683, height=0.0100099, cameraPosition=(
    0.0170434, 0.0492263, -0.497303), cameraUpVector=(0.0729927, -0.993504, 
    -0.0873033), cameraTarget=(0.00188745, 0.00429589, 0.00134856), 
    viewOffsetX=-0.00394341, viewOffsetY=0.00250599)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.482639, 
    farPlane=0.517228, width=0.0341817, height=0.013649, 
    viewOffsetX=-0.00331421, viewOffsetY=0.00406423)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.483271, 
    farPlane=0.516448, width=0.0342264, height=0.0136669, cameraPosition=(
    0.050233, 0.0455459, -0.495325), cameraUpVector=(0.0291473, -0.996353, 
    -0.0801923), cameraTarget=(0.00200144, 0.00414707, 0.0015271), 
    viewOffsetX=-0.00331855, viewOffsetY=0.00406955)
session.viewports['Viewport: 1'].view.setValues(session.views['Front'])
session.viewports['Viewport: 1'].view.setValues(session.views['Back'])
session.viewports['Viewport: 1'].view.setValues(session.views['Front'])
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0287625, 
    farPlane=0.378504, width=0.0376402, height=0.01503, cameraPosition=(
    -0.0142002, -0.199787, 0.0334216), cameraUpVector=(0.0385856, 0.124911, 
    0.991417))
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.169509, 
    farPlane=0.222112, width=0.221829, height=0.0885778, cameraPosition=(
    0.0071413, -0.0303996, -0.193205), cameraUpVector=(0.0998866, -0.981548, 
    0.163056), cameraTarget=(-0.00625612, 0.0113811, 0.0665099))
session.viewports['Viewport: 1'].view.setValues(session.views['Back'])
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0532301, 
    farPlane=0.370342, width=0.0696599, height=0.0278157, cameraPosition=(
    0.031235, -0.164202, 0.132831), cameraUpVector=(0.00308952, -0.507564, 
    -0.861608))
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.172596, 
    farPlane=0.288222, width=0.225868, height=0.0901909, cameraPosition=(
    0.0251451, -0.0614257, 0.221856), cameraUpVector=(0.0531961, -0.925298, 
    -0.375491), cameraTarget=(0.00149114, 0.0360476, -0.0216939))
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.187567, 
    farPlane=0.299058, width=0.24546, height=0.0980141, cameraPosition=(
    0.0143915, 0.0642567, 0.235426), cameraUpVector=(0.00369383, -0.984488, 
    0.175413), cameraTarget=(0.00303067, 0.0180544, -0.0236366))
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.211344, 
    farPlane=0.275281, width=0.058886, height=0.0235136, 
    viewOffsetX=0.00165577, viewOffsetY=0.0159703)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.187564, 
    farPlane=0.293409, width=0.05226, height=0.0208678, cameraPosition=(
    -0.23452, -0.0528589, 0.0193745), cameraUpVector=(0.31062, -0.791621, 
    0.526167), cameraTarget=(0.0152721, 0.0249938, -0.0109577), 
    viewOffsetX=0.00146946, viewOffsetY=0.0141733)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.218587, 
    farPlane=0.26361, width=0.0609038, height=0.0243193, cameraPosition=(
    -0.0152194, 0.0476981, -0.236868), cameraUpVector=(-0.0519328, -0.992761, 
    -0.108303), cameraTarget=(-0.000327334, 0.0184082, 0.0244696), 
    viewOffsetX=0.00171251, viewOffsetY=0.0165176)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.226024, 
    farPlane=0.255402, width=0.0629758, height=0.0251467, cameraPosition=(
    0.0679899, 0.0270544, -0.230308), cameraUpVector=(-0.0267454, -0.99905, 
    -0.0344049), cameraTarget=(-0.00844036, 0.0204222, 0.0216675), 
    viewOffsetX=0.00177077, viewOffsetY=0.0170796)
session.viewports['Viewport: 1'].view.setValues(session.views['Front'])
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.256562, 
    farPlane=0.270228, width=0.0409604, height=0.0163558, 
    viewOffsetX=0.00102716, viewOffsetY=-0.0568821)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    angle=196)
session.viewports['Viewport: 1'].view.setValues(session.views['Back'])
session.viewports['Viewport: 1'].view.setValues(session.views['Front'])
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    showModelAboveCut=True)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Z-Plane'].setValues(
    showModelAboveCut=True)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    showModelBelowCut=False)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Z-Plane'].setValues(
    showModelBelowCut=False)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.25095, 
    farPlane=0.261782, width=0.0317854, height=0.0126921, 
    viewOffsetX=-0.00147105, viewOffsetY=-0.0622282)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    angle=208)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.249924, 
    farPlane=0.262809, width=0.0433093, height=0.0172937, 
    viewOffsetX=-0.00179851, viewOffsetY=-0.0619135)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.126567, 
    farPlane=0.38412, width=0.0219328, height=0.00875792, cameraPosition=(
    -0.0442322, 0.25988, -0.00939667), cameraUpVector=(0.022307, 0.212164, 
    -0.976979), cameraTarget=(0.00139393, 0.0131388, -0.0619382), 
    viewOffsetX=-0.000910803, viewOffsetY=-0.0313544)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.232167, 
    farPlane=0.278038, width=0.0402322, height=0.016065, cameraPosition=(
    0.0224035, 0.10063, 0.241957), cameraUpVector=(0.0550367, 0.986712, 
    -0.152874), cameraTarget=(0.00322767, 0.0625341, -0.0108366), 
    viewOffsetX=-0.00167072, viewOffsetY=-0.0575146)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.246265, 
    farPlane=0.264231, width=0.0426752, height=0.0170405, cameraPosition=(
    0.0100333, 0.0683405, 0.253816), cameraUpVector=(0.0801681, 0.996555, 
    -0.0212453), cameraTarget=(0.00488668, 0.0632917, -0.0024486), 
    viewOffsetX=-0.00177217, viewOffsetY=-0.0610071)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    angle=209)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.250854, 
    farPlane=0.259642, width=0.00817798, height=0.00326553, 
    viewOffsetX=-0.00249963, viewOffsetY=-0.0624393)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.153348, 
    farPlane=0.358148, width=0.00499924, height=0.00199623, cameraPosition=(
    -0.103749, 0.229262, 0.0808108), cameraUpVector=(-0.301891, 0.441976, 
    -0.844701), cameraTarget=(-0.0192553, 0.0285224, -0.0544213), 
    viewOffsetX=-0.00152804, viewOffsetY=-0.0381694)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.149902, 
    farPlane=0.360864, width=0.00488691, height=0.00195138, cameraPosition=(
    -0.0730357, 0.239124, 0.0847385), cameraUpVector=(-0.342136, 0.456264, 
    -0.821441), cameraTarget=(-0.0218372, 0.0291817, -0.0531978), 
    viewOffsetX=-0.00149371, viewOffsetY=-0.0373118)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.148568, 
    farPlane=0.362199, width=0.0150649, height=0.00601553, 
    viewOffsetX=-0.00238008, viewOffsetY=-0.0367317)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.203091, 
    farPlane=0.310977, width=0.0205936, height=0.00822317, cameraPosition=(
    -0.12277, 0.154753, 0.176428), cameraUpVector=(0.0490731, 0.911017, 
    -0.409439), cameraTarget=(0.00295291, 0.0575668, -0.0247486), 
    viewOffsetX=-0.00325355, viewOffsetY=-0.0502119)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.202643, 
    farPlane=0.311425, width=0.017067, height=0.00681498, 
    viewOffsetX=-0.00286967, viewOffsetY=-0.0499497)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.24897, 
    farPlane=0.26423, width=0.0209688, height=0.00837299, cameraPosition=(
    -0.0330331, 0.0536092, 0.256286), cameraUpVector=(0.187873, 0.980352, 
    0.0601073), cameraTarget=(0.0123059, 0.0603864, 0.00405244), 
    viewOffsetX=-0.00352572, viewOffsetY=-0.0613689)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.248997, 
    farPlane=0.2629, width=0.020971, height=0.00837388, cameraPosition=(
    0.0170455, 0.0549597, 0.256959), cameraUpVector=(0.125114, 0.991927, 
    0.0206647), cameraTarget=(0.0084782, 0.0613774, 0.000817482), 
    viewOffsetX=-0.0035261, viewOffsetY=-0.0613755)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.250494, 
    farPlane=0.261401, width=0.0100405, height=0.00400926, 
    viewOffsetX=-0.00289572, viewOffsetY=-0.0612689)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.210462, 
    farPlane=0.30353, width=0.00843592, height=0.00336853, cameraPosition=(
    -0.14892, 0.140466, 0.167677), cameraUpVector=(0.0822586, 0.936367, 
    -0.341248), cameraTarget=(0.00547616, 0.0585532, -0.019874), 
    viewOffsetX=-0.00243295, viewOffsetY=-0.0514773)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.205267, 
    farPlane=0.308721, width=0.00822768, height=0.00328538, cameraPosition=(
    -0.154148, 0.150047, 0.154158), cameraUpVector=(0.103319, 0.916607, 
    -0.386208), cameraTarget=(0.00671777, 0.057414, -0.0226604), 
    viewOffsetX=-0.00237289, viewOffsetY=-0.0502066)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    origin=(-0.00205, 0.0007, 0))
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    origin=(-0.0021, 0.0007, 0))
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    origin=(-0.0022, 0.0007, 0))
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.204252, 
    farPlane=0.309736, width=0.0163696, height=0.00653652, 
    viewOffsetX=-0.00212386, viewOffsetY=-0.0498516)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.185033, 
    farPlane=0.329741, width=0.0148293, height=0.00592146, cameraPosition=(
    -0.0116661, -0.0918594, 0.247656), cameraUpVector=(-0.272428, 0.799682, 
    0.535062), cameraTarget=(-0.0159371, 0.0496791, 0.0339457), 
    viewOffsetX=-0.00192401, viewOffsetY=-0.0451608)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.201568, 
    farPlane=0.311869, width=0.0161545, height=0.00645061, cameraPosition=(
    -0.114429, 0.156113, 0.180395), cameraUpVector=(-0.108749, 0.872765, 
    -0.475872), cameraTarget=(-0.00626844, 0.0553788, -0.029074), 
    viewOffsetX=-0.00209594, viewOffsetY=-0.0491965)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.234686, 
    farPlane=0.278832, width=0.0188088, height=0.00751047, cameraPosition=(
    -0.0778127, 0.0959966, 0.233654), cameraUpVector=(0.084529, 0.990126, 
    -0.111827), cameraTarget=(0.00598528, 0.0617541, -0.00619696), 
    viewOffsetX=-0.00244031, viewOffsetY=-0.0572797)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.244618, 
    farPlane=0.268482, width=0.0196048, height=0.0078283, cameraPosition=(
    -0.0192786, 0.0454559, 0.259284), cameraUpVector=(0.105491, 0.991665, 
    0.0739701), cameraTarget=(0.00751366, 0.0615865, 0.00483359), 
    viewOffsetX=-0.00254358, viewOffsetY=-0.0597037)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.228819, 
    farPlane=0.283788, width=0.0183386, height=0.0073227, cameraPosition=(
    0.0291774, 0.0138016, 0.261667), cameraUpVector=(0.00168344, 0.982479, 
    0.186366), cameraTarget=(0.0010668, 0.0613387, 0.0113214), 
    viewOffsetX=-0.0023793, viewOffsetY=-0.0558476)
session.viewports['Viewport: 1'].view.setValues(session.views['Back'])
session.viewports['Viewport: 1'].view.setValues(session.views['Front'])
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.256607, 
    farPlane=0.271138, width=0.0437618, height=0.0174744, 
    viewOffsetX=0.0028484, viewOffsetY=0.0573811)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM5', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].view.setValues(session.views['Top'])
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0770627, 
    farPlane=0.330247, width=0.00541208, height=0.00216109, 
    viewOffsetX=-0.0012952, viewOffsetY=0.000126931)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM1', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='SDV_AR49_THETA_COVERAGE', outputPosition=INTEGRATION_POINT, 
    )
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
session.viewports[session.currentViewportName].odbDisplay.setFrame(
    step='step2_insitu', frame=56)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0773548, 
    farPlane=0.33063, width=0.00891219, height=0.00355871, 
    viewOffsetX=-0.00123007, viewOffsetY=-0.000228651)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM1', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0771077, 
    farPlane=0.330877, width=0.00948943, height=0.0037892, 
    viewOffsetX=-0.00110673, viewOffsetY=-0.000174287)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.160938, 
    farPlane=0.249993, width=0.0198062, height=0.00790875, cameraPosition=(
    -0.0745586, 0.0662924, 0.179771), cameraUpVector=(-0.0194639, 0.936787, 
    -0.349359), cameraTarget=(0.0216045, -0.0178139, -0.0511124), 
    viewOffsetX=-0.00230994, viewOffsetY=-0.000363768)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.19364, 
    farPlane=0.215828, width=0.0238307, height=0.00951578, cameraPosition=(
    -0.0102954, 0.0116257, 0.204169), cameraUpVector=(0.257587, 0.965304, 
    -0.0428515), cameraTarget=(0.00363, -0.00376782, -0.0588848), 
    viewOffsetX=-0.00277931, viewOffsetY=-0.000437684)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.130819, 
    farPlane=0.281001, width=0.0160995, height=0.00642864, cameraPosition=(
    -0.123124, 0.11045, 0.123016), cameraUpVector=(0.050444, 0.769235, 
    -0.636971), cameraTarget=(0.0354638, -0.0301302, -0.0341953), 
    viewOffsetX=-0.00187764, viewOffsetY=-0.000295689)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.196949, 
    farPlane=0.2144, width=0.0242379, height=0.00967835, cameraPosition=(
    -0.0650377, 0.00553015, 0.195057), cameraUpVector=(0.0192227, 0.999552, 
    -0.0229556), cameraTarget=(0.0196596, -0.00183597, -0.0547436), 
    viewOffsetX=-0.0028268, viewOffsetY=-0.000445161)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.196222, 
    farPlane=0.214139, width=0.0241484, height=0.00964261, cameraPosition=(
    -0.0134462, 0.00714315, 0.204632), cameraUpVector=(-0.0329325, 0.998754, 
    -0.0374966), cameraTarget=(0.00521987, -0.00211746, -0.0584156), 
    viewOffsetX=-0.00281636, viewOffsetY=-0.000443517)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    origin=(-0.0022, 0.0005, 0))
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    origin=(-0.0022, 0.0003, 0))
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.197313, 
    farPlane=0.213047, width=0.0157468, height=0.0062878, 
    viewOffsetX=-0.00297212, viewOffsetY=-0.000237661)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.198607, 
    farPlane=0.211767, width=0.01585, height=0.00632903, cameraPosition=(
    -0.00883133, -0.00570339, 0.204909), cameraUpVector=(-0.0237116, 0.999375, 
    0.0261997), cameraTarget=(0.00390909, 0.00150551, -0.0585564), 
    viewOffsetX=-0.00299161, viewOffsetY=-0.000239219)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.196896, 
    farPlane=0.213478, width=0.0258835, height=0.0103355, 
    viewOffsetX=-0.00374495, viewOffsetY=-0.00100753)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    angle=202)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.185567, 
    farPlane=0.224398, width=0.0243941, height=0.00974076, cameraPosition=(
    -0.0112292, 0.0239038, 0.203326), cameraUpVector=(-0.00569391, 0.993028, 
    -0.11774), cameraTarget=(0.00461736, -0.00701956, -0.0582476), 
    viewOffsetX=-0.00352946, viewOffsetY=-0.000949556)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM8', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.167935, 
    farPlane=0.244171, width=0.0220763, height=0.00881524, cameraPosition=(
    -0.106378, 0.0517431, 0.16885), cameraUpVector=(-0.528958, 0.665234, 
    -0.526942), cameraTarget=(0.02962, -0.0119781, -0.0481118), 
    viewOffsetX=-0.00319411, viewOffsetY=-0.000859334)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.101457, 
    farPlane=0.30793, width=0.0133372, height=0.00532566, cameraPosition=(
    -0.0539448, 0.160453, 0.115671), cameraUpVector=(0.0382986, 0.595935, 
    -0.802119), cameraTarget=(0.0171189, -0.0451462, -0.0336856), 
    viewOffsetX=-0.0019297, viewOffsetY=-0.00051916)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0951398, 
    farPlane=0.311623, width=0.0125068, height=0.00499406, cameraPosition=(
    0.0456105, 0.170459, 0.101852), cameraUpVector=(-0.288029, 0.553579, 
    -0.781402), cameraTarget=(-0.0122551, -0.0491959, -0.0324304), 
    viewOffsetX=-0.00180955, viewOffsetY=-0.000486835)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0984626, 
    farPlane=0.310015, width=0.0129436, height=0.00516848, cameraPosition=(
    -0.0282595, 0.168662, 0.112311), cameraUpVector=(-0.294184, 0.501829, 
    -0.813402), cameraTarget=(0.00977081, -0.0471963, -0.0346165), 
    viewOffsetX=-0.00187275, viewOffsetY=-0.000503838)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0856438, 
    farPlane=0.322765, width=0.0112585, height=0.0044956, cameraPosition=(
    -0.0304006, 0.18867, 0.0730735), cameraUpVector=(0.0332107, 0.369644, 
    -0.92858), cameraTarget=(0.0110499, -0.0539551, -0.0220259), 
    viewOffsetX=-0.00162894, viewOffsetY=-0.000438243)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0776219, 
    farPlane=0.330512, width=0.010204, height=0.00407451, cameraPosition=(
    -0.0226975, 0.202228, 0.0201041), cameraUpVector=(0.0391457, 0.107388, 
    -0.993446), cameraTarget=(0.00883235, -0.0583672, -0.00682205), 
    viewOffsetX=-0.00147636, viewOffsetY=-0.000397194)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0786988, 
    farPlane=0.329435, width=0.00362829, height=0.0014488, 
    viewOffsetX=-0.0015622, viewOffsetY=-0.000543526)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM10', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0783423, 
    farPlane=0.329792, width=0.00670581, height=0.00267768, 
    viewOffsetX=-0.00123256, viewOffsetY=-0.000726105)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.142963, 
    farPlane=0.266776, width=0.0122371, height=0.00488635, cameraPosition=(
    -0.00812484, 0.100292, 0.178704), cameraUpVector=(-0.218717, 0.851156, 
    -0.477176), cameraTarget=(0.00385893, -0.0262643, -0.0525292), 
    viewOffsetX=-0.00224924, viewOffsetY=-0.00132503)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.14686, 
    farPlane=0.263299, width=0.0125706, height=0.00501953, cameraPosition=(
    -0.0317067, 0.0916914, 0.180903), cameraUpVector=(-0.203683, 0.862753, 
    -0.462786), cameraTarget=(0.0106503, -0.0235878, -0.0526468), 
    viewOffsetX=-0.00231054, viewOffsetY=-0.00136114)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    angle=194)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM5', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    angle=210)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.147645, 
    farPlane=0.262513, width=0.00727103, height=0.00290338, 
    viewOffsetX=-0.00248167, viewOffsetY=-0.00108762)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.194375, 
    farPlane=0.215171, width=0.0095723, height=0.00382229, cameraPosition=(
    0.0303314, -0.0104544, 0.202233), cameraUpVector=(-0.247057, 0.964219, 
    0.0961488), cameraTarget=(-0.00725202, 0.00590757, -0.0584363), 
    viewOffsetX=-0.00326711, viewOffsetY=-0.00143185)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.185705, 
    farPlane=0.224011, width=0.00914531, height=0.00365179, cameraPosition=(
    0.00911595, 0.0290597, 0.20265), cameraUpVector=(0.0187519, 0.9905, 
    -0.136225), cameraTarget=(-0.000431863, -0.00669291, -0.0586149), 
    viewOffsetX=-0.00312138, viewOffsetY=-0.00136798)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.184893, 
    farPlane=0.224669, width=0.00910534, height=0.00363583, cameraPosition=(
    0.0177193, 0.0298561, 0.201887), cameraUpVector=(0.00997808, 0.990026, 
    -0.140532), cameraTarget=(-0.00293749, -0.00691152, -0.0585934), 
    viewOffsetX=-0.00310774, viewOffsetY=-0.001362)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.186539, 
    farPlane=0.223599, width=0.0091864, height=0.0036682, cameraPosition=(
    -0.0151376, 0.0276092, 0.202703), cameraUpVector=(0.0291913, 0.991562, 
    -0.126301), cameraTarget=(0.0065779, -0.00624954, -0.0580852), 
    viewOffsetX=-0.00313541, viewOffsetY=-0.00137413)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.185679, 
    farPlane=0.224459, width=0.0141008, height=0.00563056, 
    viewOffsetX=-0.00319516, viewOffsetY=-0.00143969)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    angle=200)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.164724, 
    farPlane=0.244504, width=0.0125094, height=0.00499511, cameraPosition=(
    0.0246348, 0.0610211, 0.193885), cameraUpVector=(-0.0351489, 0.956669, 
    -0.289049), cameraTarget=(-0.00497256, -0.0158136, -0.056811), 
    viewOffsetX=-0.00283456, viewOffsetY=-0.00127721)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.183689, 
    farPlane=0.226752, width=0.0139496, height=0.0055702, cameraPosition=(
    -0.0361914, 0.03078, 0.199721), cameraUpVector=(-0.0273552, 0.988345, 
    -0.149752), cameraTarget=(0.0126108, -0.00674847, -0.0568698), 
    viewOffsetX=-0.00316091, viewOffsetY=-0.00142426)
session.viewports['Viewport: 1'].odbDisplay.setValues(viewCut=OFF)
session.viewports['Viewport: 1'].odbDisplay.setValues(viewCutNames=('Y-Plane', 
    ), viewCut=ON)
session.viewports['Viewport: 1'].odbDisplay.setValues(viewCut=OFF)
session.viewports['Viewport: 1'].odbDisplay.setValues(viewCutNames=('Y-Plane', 
    ), viewCut=ON)
session.viewports['Viewport: 1'].odbDisplay.setValues(viewCut=OFF)
session.viewports['Viewport: 1'].odbDisplay.setValues(viewCutNames=('Y-Plane', 
    ), viewCut=ON)
session.viewports['Viewport: 1'].odbDisplay.setValues(viewCut=OFF)
session.viewports['Viewport: 1'].odbDisplay.setValues(viewCutNames=('Y-Plane', 
    ), viewCut=ON)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    angle=180)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.106025, 
    farPlane=0.302637, width=0.00805174, height=0.00321512, cameraPosition=(
    0.0392251, 0.154512, 0.128346), cameraUpVector=(0.249941, 0.585344, 
    -0.771299), cameraTarget=(-0.00851728, -0.0445319, -0.0381798), 
    viewOffsetX=-0.00182448, viewOffsetY=-0.000822083)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.106881, 
    farPlane=0.30178, width=0.00811675, height=0.00324108, cameraPosition=(
    0.0390667, 0.154938, 0.127882), cameraUpVector=(0.0868205, 0.626922, 
    -0.77423), cameraTarget=(-0.00867568, -0.0441059, -0.0386435), 
    viewOffsetX=-0.00183921, viewOffsetY=-0.00082872)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0807813, 
    farPlane=0.328617, width=0.00613469, height=0.00244963, cameraPosition=(
    -0.029103, 0.199329, 0.0386692), cameraUpVector=(-0.09298, 0.18376, 
    -0.978564), cameraTarget=(0.0108412, -0.0563076, -0.0131301), 
    viewOffsetX=-0.00139009, viewOffsetY=-0.000626351)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0810701, 
    farPlane=0.328329, width=0.00615663, height=0.00245839, cameraPosition=(
    -0.0295405, 0.199387, 0.0380441), cameraUpVector=(-0.285111, 0.147363, 
    -0.947099), cameraTarget=(0.0104037, -0.0562493, -0.0137552), 
    viewOffsetX=-0.00139506, viewOffsetY=-0.000628591)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    angle=213)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    angle=180)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.127332, 
    farPlane=0.283477, width=0.00966982, height=0.00386123, cameraPosition=(
    -0.0978025, 0.121438, 0.134113), cameraUpVector=(0.186245, 0.796243, 
    -0.575595), cameraTarget=(0.0306781, -0.0325707, -0.0373592), 
    viewOffsetX=-0.00219113, viewOffsetY=-0.000987287)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.199579, 
    farPlane=0.211249, width=0.0151564, height=0.00605206, cameraPosition=(
    -0.0387867, 0.00484028, 0.201688), cameraUpVector=(0.125909, 0.992022, 
    0.00624931), cameraTarget=(0.0142921, -0.000269535, -0.0567406), 
    viewOffsetX=-0.00343436, viewOffsetY=-0.00154747)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.176868, 
    farPlane=0.234245, width=0.0134317, height=0.00536337, cameraPosition=(
    -0.0367412, -0.0394113, 0.198303), cameraUpVector=(0.0876808, 0.972561, 
    0.215493), cameraTarget=(0.01366, 0.01228, -0.0555009), 
    viewOffsetX=-0.00304355, viewOffsetY=-0.00137138)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.109368, 
    farPlane=0.301367, width=0.00830561, height=0.00331649, cameraPosition=(
    -0.0891035, 0.15003, 0.108901), cameraUpVector=(0.0929315, 0.627209, 
    -0.773287), cameraTarget=(0.0281325, -0.0402652, -0.0313565), 
    viewOffsetX=-0.00188201, viewOffsetY=-0.000848005)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.095317, 
    farPlane=0.315784, width=0.00723855, height=0.00289041, cameraPosition=(
    -0.110625, 0.173627, 0.00451861), cameraUpVector=(-0.0201956, 0.0238734, 
    -0.999511), cameraTarget=(0.0337853, -0.0470709, -0.00367001), 
    viewOffsetX=-0.00164022, viewOffsetY=-0.000739058)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0799413, 
    farPlane=0.329696, width=0.00607089, height=0.00242415, cameraPosition=(
    0.00510731, 0.202688, 0.0319587), cameraUpVector=(-0.198184, 0.167168, 
    -0.965804), cameraTarget=(0.00123438, -0.0574236, -0.0122677), 
    viewOffsetX=-0.00137563, viewOffsetY=-0.00061984)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0788015, 
    farPlane=0.331101, width=0.00598433, height=0.00238959, cameraPosition=(
    -0.0180635, 0.204215, 0.012445), cameraUpVector=(-0.0503015, 0.0651433, 
    -0.996607), cameraTarget=(0.00827978, -0.0576912, -0.00600338), 
    viewOffsetX=-0.00135602, viewOffsetY=-0.000611002)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    angle=207)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.151566, 
    farPlane=0.258538, width=0.0115102, height=0.00459612, cameraPosition=(
    -0.0507129, 0.0838537, 0.180337), cameraUpVector=(0.114737, 0.91631, 
    -0.383681), cameraTarget=(0.0178453, -0.0218198, -0.0515299), 
    viewOffsetX=-0.00260816, viewOffsetY=-0.0011752)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0979142, 
    farPlane=0.313065, width=0.00743579, height=0.00296917, cameraPosition=(
    -0.115458, 0.1685, 0.0255429), cameraUpVector=(0.315066, 0.358489, 
    -0.87876), cameraTarget=(0.0359076, -0.0451273, -0.00733584), 
    viewOffsetX=-0.00168492, viewOffsetY=-0.000759199)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    angle=177)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.200563, 
    farPlane=0.209343, width=0.0152311, height=0.00608191, cameraPosition=(
    0.0740048, 0.00204378, 0.191143), cameraUpVector=(-0.242677, 0.965696, 
    0.0924107), cameraTarget=(-0.0183892, 0.00247734, -0.0560251), 
    viewOffsetX=-0.00345131, viewOffsetY=-0.00155511)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.194731, 
    farPlane=0.215823, width=0.0147882, height=0.00590507, cameraPosition=(
    -0.0305948, -0.00974478, 0.202756), cameraUpVector=(0.119365, 0.990175, 
    0.0728419), cameraTarget=(0.0127599, 0.00414908, -0.0571599), 
    viewOffsetX=-0.00335096, viewOffsetY=-0.00150989)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.098743, 
    farPlane=0.310867, width=0.00749871, height=0.00299431, cameraPosition=(
    0.0111412, 0.169615, 0.114911), cameraUpVector=(0.12071, 0.557773, 
    -0.821169), cameraTarget=(0.000907163, -0.0477998, -0.0342706), 
    viewOffsetX=-0.00169918, viewOffsetY=-0.000765626)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.200514, 
    farPlane=0.209391, width=0.0152274, height=0.00608045, cameraPosition=(
    0.00647066, -0.00282223, 0.204852), cameraUpVector=(0.0827904, 0.99636, 
    0.0203142), cameraTarget=(0.00219708, 0.00291071, -0.0589239), 
    viewOffsetX=-0.00345047, viewOffsetY=-0.00155473)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.198789, 
    farPlane=0.211115, width=0.0205701, height=0.00821379, 
    viewOffsetX=-0.00471013, viewOffsetY=-0.00147051)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0998349, 
    farPlane=0.311091, width=0.0103306, height=0.0041251, cameraPosition=(
    -0.0434937, 0.166694, 0.112639), cameraUpVector=(0.0326828, 0.572491, 
    -0.819259), cameraTarget=(0.0163194, -0.0450864, -0.0329652), 
    viewOffsetX=-0.0023655, viewOffsetY=-0.000738515)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.1846, 
    farPlane=0.226557, width=0.0191018, height=0.00762752, cameraPosition=(
    -0.044621, 0.029082, 0.198648), cameraUpVector=(0.186163, 0.977948, 
    -0.094663), cameraTarget=(0.0168213, -0.00720478, -0.0553936), 
    viewOffsetX=-0.00437393, viewOffsetY=-0.00136555)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.183323, 
    farPlane=0.227834, width=0.0189697, height=0.00757476, cameraPosition=(
    -0.044785, 0.0299205, 0.198489), cameraUpVector=(0.0214727, 0.990438, 
    -0.136279), cameraTarget=(0.0166573, -0.00636627, -0.055553), 
    viewOffsetX=-0.00434368, viewOffsetY=-0.0013561)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.194533, 
    farPlane=0.215984, width=0.0201296, height=0.00803793, cameraPosition=(
    -0.0038034, 0.0131027, 0.204858), cameraUpVector=(0.0668885, 0.99624, 
    -0.0550696), cameraTarget=(0.0052806, -0.00206056, -0.0584225), 
    viewOffsetX=-0.00460928, viewOffsetY=-0.00143902)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.194366, 
    farPlane=0.216149, width=0.0201124, height=0.00803103, cameraPosition=(
    -0.00490889, 0.0152468, 0.204696), cameraUpVector=(-0.40033, 0.913958, 
    -0.0664512), cameraTarget=(0.00417511, 8.35771e-05, -0.0585841), 
    viewOffsetX=-0.00460532, viewOffsetY=-0.00143778)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0978257, 
    farPlane=0.312658, width=0.0101227, height=0.00404207, cameraPosition=(
    -0.0294033, 0.171151, 0.110113), cameraUpVector=(-0.40692, 0.45358, 
    -0.792894), cameraTarget=(0.0110619, -0.0456916, -0.0347006), 
    viewOffsetX=-0.00231789, viewOffsetY=-0.000723645)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0779298, 
    farPlane=0.33215, width=0.00806394, height=0.00321999, cameraPosition=(
    -0.00790628, 0.205333, 0.00093633), cameraUpVector=(0.0136585, 0.0125289, 
    -0.999828), cameraTarget=(0.00635417, -0.0581368, -0.00217069), 
    viewOffsetX=-0.00184648, viewOffsetY=-0.00057647)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    angle=235)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0779879, 
    farPlane=0.332092, width=0.0103362, height=0.00412731, 
    viewOffsetX=-0.00189341, viewOffsetY=-0.000669777)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.18588, 
    farPlane=0.225483, width=0.0246357, height=0.00983722, cameraPosition=(
    -0.0480059, 0.0287109, 0.198024), cameraUpVector=(-0.0807402, 0.985175, 
    -0.151364), cameraTarget=(0.0174345, -0.00486104, -0.055392), 
    viewOffsetX=-0.00451285, viewOffsetY=-0.00159638)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    angle=360)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    angle=247)
session.viewports['Viewport: 1'].odbDisplay.viewCuts['Y-Plane'].setValues(
    angle=180)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.185208, 
    farPlane=0.226155, width=0.0140651, height=0.0056163, 
    viewOffsetX=-0.00473795, viewOffsetY=-0.00146846)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.166917, 
    farPlane=0.244087, width=0.0126761, height=0.00506165, cameraPosition=(
    -0.0296658, 0.0596604, 0.194565), cameraUpVector=(-0.0695586, 0.953978, 
    -0.291697), cameraTarget=(0.0123998, -0.0137008, -0.0553908), 
    viewOffsetX=-0.00427005, viewOffsetY=-0.00132344)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.165125, 
    farPlane=0.24588, width=0.0280308, height=0.0111929, 
    viewOffsetX=-0.00403528, viewOffsetY=-0.000985513)
session.viewports['Viewport: 1'].view.setValues(session.views['Front'])
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.244634, 
    farPlane=0.260161, width=0.0469992, height=0.0187671, 
    viewOffsetX=0.00381266, viewOffsetY=0.0537359)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM1', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM8', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.151403, 
    farPlane=0.344758, width=0.0290877, height=0.011615, cameraPosition=(
    0.0172943, 0.149174, 0.205556), cameraUpVector=(0.655822, 0.417222, 
    -0.629146), cameraTarget=(-0.0352802, -0.027765, 0.033414), 
    viewOffsetX=0.00235965, viewOffsetY=0.0332571)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.113634, 
    farPlane=0.37165, width=0.0218315, height=0.00871753, cameraPosition=(
    -0.0772074, 0.234816, 0.0310843), cameraUpVector=(0.666275, 0.0842808, 
    -0.740928), cameraTarget=(-0.0342169, -0.0136759, 0.0414771), 
    viewOffsetX=0.00177101, viewOffsetY=0.0249608)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.121189, 
    farPlane=0.367849, width=0.0232831, height=0.00929715, cameraPosition=(
    0.0705884, 0.223617, 0.090837), cameraUpVector=(0.0866329, 0.119315, 
    -0.98907), cameraTarget=(-0.00888492, -0.0132906, 0.0552968), 
    viewOffsetX=0.00188876, viewOffsetY=0.0266204)
session.viewports['Viewport: 1'].view.setValues(session.views['Front'])
session.viewports['Viewport: 1'].view.setValues(session.views['Back'])
session.viewports['Viewport: 1'].view.setValues(session.views['Top'])
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0630347, 
    farPlane=0.316225, width=0.00761536, height=0.00304087, 
    viewOffsetX=-0.000311443, viewOffsetY=9.52508e-05)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0631642, 
    farPlane=0.315987, width=0.007631, height=0.00304712, cameraPosition=(
    0.00675503, 0.189363, 0.0138862), cameraUpVector=(-0.0343915, 0.0746393, 
    -0.996617), cameraTarget=(-0.00222433, -0.062193, -0.00464367), 
    viewOffsetX=-0.000312083, viewOffsetY=9.54465e-05)
session.viewports['Viewport: 1'].view.setValues(session.views['Top'])
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0635318, 
    farPlane=0.315728, width=0.00439798, height=0.00175615, 
    viewOffsetX=-0.00137612, viewOffsetY=8.64034e-05)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM1', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM5', outputPosition=INTEGRATION_POINT, )
o3 = session.odbs['C:/LocalUserData/User-data/nguyenb5/CP1000 plastic (UMAT UMATHT)/CP1000_dense_CSC/CP1000_NDBR40_dense_CSC/NDBR40_combined_processed.odb']
session.viewports['Viewport: 1'].setValues(displayedObject=o3)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.150674, 
    farPlane=0.240807, width=0.0660376, height=0.0263693, 
    viewOffsetX=0.000246843, viewOffsetY=0.00263188)
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.149043, 
    farPlane=0.241876, width=0.0655898, height=0.0261905, 
    viewOffsetX=0.000857732, viewOffsetY=0.00299544)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM8', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.155462, 
    farPlane=0.235456, width=0.0238959, height=0.00954183, 
    viewOffsetX=-6.79292e-05, viewOffsetY=0.00190023)
session.viewports[session.currentViewportName].odbDisplay.setFrame(
    step='step2_insitu', frame=14)
session.viewports[session.currentViewportName].odbDisplay.setFrame(
    step='step2_insitu', frame=102)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='SDV_AR1_SIG11', outputPosition=INTEGRATION_POINT, )
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
#: Warning: The selected Primary Variable is not available in the current step/frame.
session.viewports[session.currentViewportName].odbDisplay.setFrame(
    step='step2_insitu', frame=58)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.156048, 
    farPlane=0.235109, width=0.0217726, height=0.00869398, 
    viewOffsetX=-0.000468651, viewOffsetY=0.00170669)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM1', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM8', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.149852, 
    farPlane=0.241304, width=0.0619992, height=0.0247568, 
    viewOffsetX=0.0017797, viewOffsetY=0.00161422)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='SDV_AR33_TRIAX', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='SDV_AR34_LODE', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='SDV_AR44_THETAL', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.150265, 
    farPlane=0.240892, width=0.0664089, height=0.0265176, 
    viewOffsetX=0.00468483, viewOffsetY=0.000968137)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='SDV_AR45_THETAT_DIS', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='SDV_AR46_THETAT_GB', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='SDV_AR45_THETAT_DIS', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.142828, 
    farPlane=0.260469, width=0.0631221, height=0.0252052, cameraPosition=(
    0.191039, -0.0610774, -0.0209214), cameraUpVector=(0.0545061, 0.490038, 
    -0.869995), cameraTarget=(-0.0502228, 0.0256697, 0.0128246), 
    viewOffsetX=0.00445297, viewOffsetY=0.000920222)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.143273, 
    farPlane=0.260023, width=0.0633187, height=0.0252837, cameraPosition=(
    0.190278, -0.0639517, -0.0189711), cameraUpVector=(0.235509, 0.843082, 
    -0.483475), cameraTarget=(-0.0509835, 0.0227954, 0.0147749), 
    viewOffsetX=0.00446684, viewOffsetY=0.000923087)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.170168, 
    farPlane=0.230041, width=0.0752048, height=0.0300299, cameraPosition=(
    -0.00423795, -0.0315923, -0.197573), cameraUpVector=(0.0945606, 0.982172, 
    -0.162473), cameraTarget=(0.0100027, 0.0092126, 0.0573838), 
    viewOffsetX=0.00530535, viewOffsetY=0.00109637)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.155386, 
    farPlane=0.246783, width=0.0686721, height=0.0274213, cameraPosition=(
    0.154609, 0.0462938, -0.120344), cameraUpVector=(-0.146497, 0.972232, 
    0.18249), cameraTarget=(-0.0388208, -0.0130778, 0.0406856), 
    viewOffsetX=0.0048445, viewOffsetY=0.00100113)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.130955, 
    farPlane=0.269799, width=0.0578748, height=0.0231098, cameraPosition=(
    0.139134, -0.0866369, -0.115008), cameraUpVector=(0.387283, 0.897981, 
    -0.208909), cameraTarget=(-0.0354411, 0.0246949, 0.039909), 
    viewOffsetX=0.0040828, viewOffsetY=0.000843722)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.182065, 
    farPlane=0.219485, width=0.0804624, height=0.0321292, cameraPosition=(
    0.181085, -0.00616557, -0.0866764), cameraUpVector=(0.0843186, 0.990924, 
    0.104689), cameraTarget=(-0.0485422, 0.000832306, 0.032036), 
    viewOffsetX=0.00567625, viewOffsetY=0.00117301)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.176741, 
    farPlane=0.224807, width=0.100044, height=0.0399485, 
    viewOffsetX=0.00425566, viewOffsetY=0.00216805)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.166394, 
    farPlane=0.233456, width=0.0941874, height=0.0376098, cameraPosition=(
    0.0779611, 0.0238685, -0.182704), cameraUpVector=(0.217067, 0.952064, 
    0.215535), cameraTarget=(-0.0164438, -0.00861043, 0.0558395), 
    viewOffsetX=0.00400651, viewOffsetY=0.00204112)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.177473, 
    farPlane=0.219682, width=0.100459, height=0.040114, cameraPosition=(
    -0.143169, 1.84174e-05, -0.137734), cameraUpVector=(-0.31193, 0.891945, 
    0.327313), cameraTarget=(0.0481977, 0.00313005, 0.0361608), 
    viewOffsetX=0.00427328, viewOffsetY=0.00217702)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.164854, 
    farPlane=0.231645, width=0.0933159, height=0.0372617, cameraPosition=(
    -0.19678, -0.0195088, -0.0139938), cameraUpVector=(-0.0627166, 0.918916, 
    -0.389435), cameraTarget=(0.060415, 0.00374042, -0.000555735), 
    viewOffsetX=0.00396943, viewOffsetY=0.00202222)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.134918, 
    farPlane=0.260647, width=0.0763705, height=0.0304953, cameraPosition=(
    0.000813709, -0.0775038, -0.181767), cameraUpVector=(0.175918, 0.90621, 
    -0.384495), cameraTarget=(0.00512729, 0.0227754, 0.0565521), 
    viewOffsetX=0.00324862, viewOffsetY=0.001655)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.103725, 
    farPlane=0.294596, width=0.0587135, height=0.0234447, cameraPosition=(
    0.0969899, 0.128121, -0.118378), cameraUpVector=(-0.579885, 0.742157, 
    0.33606), cameraTarget=(-0.024022, -0.0376596, 0.0389231), 
    viewOffsetX=0.00249753, viewOffsetY=0.00127236)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.169095, 
    farPlane=0.227959, width=0.095716, height=0.03822, cameraPosition=(
    0.182879, 0.0198637, 0.0749953), cameraUpVector=(-0.165655, 0.977342, 
    0.13176), cameraTarget=(-0.057191, -0.00844473, -0.0168499), 
    viewOffsetX=0.00407153, viewOffsetY=0.00207423)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0869448, 
    farPlane=0.311887, width=0.049215, height=0.0196519, cameraPosition=(
    -0.0389637, 0.152984, 0.122682), cameraUpVector=(0.163315, 0.634851, 
    -0.755178), cameraTarget=(0.00490635, -0.0466821, -0.0356828), 
    viewOffsetX=0.00209349, viewOffsetY=0.00106652)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.112044, 
    farPlane=0.286932, width=0.0634225, height=0.0253251, cameraPosition=(
    0.150312, 0.118597, -0.0574522), cameraUpVector=(-0.647914, 0.754361, 
    -0.105583), cameraTarget=(-0.0415656, -0.0349381, 0.0230485), 
    viewOffsetX=0.00269784, viewOffsetY=0.00137441)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.111221, 
    farPlane=0.287539, width=0.0629564, height=0.025139, cameraPosition=(
    0.108974, 0.116372, 0.120404), cameraUpVector=(-0.52261, 0.794296, 
    -0.309795), cameraTarget=(-0.0348114, -0.0369719, -0.0302006), 
    viewOffsetX=0.00267801, viewOffsetY=0.00136431)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.181326, 
    farPlane=0.217277, width=0.102639, height=0.0409846, cameraPosition=(
    0.0165314, 0.0110039, 0.198414), cameraUpVector=(0.156176, 0.984769, 
    -0.0764076), cameraTarget=(-0.0115989, -0.00444403, -0.0581795), 
    viewOffsetX=0.00436601, viewOffsetY=0.00222426)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.158844, 
    farPlane=0.239697, width=0.0899134, height=0.0359032, cameraPosition=(
    0.177174, 0.0290629, 0.0868103), cameraUpVector=(-0.0197107, 0.953942, 
    -0.299343), cameraTarget=(-0.055436, -0.00910926, -0.0195188), 
    viewOffsetX=0.0038247, viewOffsetY=0.00194849)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.120106, 
    farPlane=0.278559, width=0.0679858, height=0.0271473, cameraPosition=(
    0.114895, 0.0956031, 0.132407), cameraUpVector=(-0.405779, 0.867455, 
    -0.287863), cameraTarget=(-0.0373074, -0.0308416, -0.0340758), 
    viewOffsetX=0.00289195, viewOffsetY=0.0014733)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.1774, 
    farPlane=0.22116, width=0.100417, height=0.0400973, cameraPosition=(
    0.100547, 0.010577, 0.171848), cameraUpVector=(0.00701973, 0.99711, 
    -0.0756513), cameraTarget=(-0.0352956, -0.00511887, -0.0476303), 
    viewOffsetX=0.00427149, viewOffsetY=0.0021761)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.170848, 
    farPlane=0.227762, width=0.0967082, height=0.0386163, cameraPosition=(
    0.0063167, 0.0196268, 0.198365), cameraUpVector=(0.0418652, 0.993221, 
    -0.108441), cameraTarget=(-0.00827287, -0.00778774, -0.0583562), 
    viewOffsetX=0.00411372, viewOffsetY=0.00209573)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.175579, 
    farPlane=0.22299, width=0.0993863, height=0.0396857, cameraPosition=(
    0.0828459, 0.00668799, 0.181224), cameraUpVector=(-0.2084, 0.976789, 
    0.049531), cameraTarget=(-0.0294893, -0.0054847, -0.0513764), 
    viewOffsetX=0.00422764, viewOffsetY=0.00215377)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM1', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.173415, 
    farPlane=0.225155, width=0.118184, height=0.0471916, 
    viewOffsetX=0.00611952, viewOffsetY=0.00113639)
session.Viewport(name='Viewport: 2', origin=(5.375, 46.0162048339844), 
    width=235.604156494141, height=99.1805572509766)
session.viewports['Viewport: 2'].makeCurrent()
session.viewports['Viewport: 2'].maximize()
session.viewports['Viewport: 1'].restore()
session.Viewport(name='Viewport: 3', origin=(10.75, 46.0162048339844), 
    width=235.604156494141, height=93.8194427490234)
session.viewports['Viewport: 3'].makeCurrent()
session.viewports['Viewport: 3'].maximize()
session.viewports['Viewport: 2'].restore()
session.Viewport(name='Viewport: 4', origin=(16.125, 46.0162048339844), 
    width=235.604156494141, height=88.4583358764648)
session.viewports['Viewport: 4'].makeCurrent()
session.viewports['Viewport: 4'].maximize()
session.viewports['Viewport: 3'].restore()
session.Viewport(name='Viewport: 5', origin=(21.5, 46.0162048339844), 
    width=235.604156494141, height=83.0972213745117)
session.viewports['Viewport: 5'].makeCurrent()
session.viewports['Viewport: 5'].maximize()
session.viewports['Viewport: 4'].restore()
session.Viewport(name='Viewport: 6', origin=(0.0, 46.0162048339844), 
    width=235.604156494141, height=104.541664123535)
session.viewports['Viewport: 6'].makeCurrent()
session.viewports['Viewport: 6'].maximize()
session.viewports['Viewport: 5'].restore()
session.Viewport(name='Viewport: 7', origin=(5.375, 46.0162048339844), 
    width=235.604156494141, height=99.1805572509766)
session.viewports['Viewport: 7'].makeCurrent()
session.viewports['Viewport: 7'].maximize()
session.viewports['Viewport: 6'].restore()
session.Viewport(name='Viewport: 8', origin=(10.75, 46.0162048339844), 
    width=235.604156494141, height=93.8194427490234)
session.viewports['Viewport: 8'].makeCurrent()
session.viewports['Viewport: 8'].maximize()
session.viewports['Viewport: 7'].restore()
session.Viewport(name='Viewport: 9', origin=(16.125, 46.0162048339844), 
    width=235.604156494141, height=88.4583358764648)
session.viewports['Viewport: 9'].makeCurrent()
session.viewports['Viewport: 9'].maximize()
session.viewports['Viewport: 8'].restore()
session.viewports['Viewport: 9'].restore()
session.viewports['Viewport: 1'].setValues(origin=(0.0, 115.710639953613), 
    width=87.1197891235352, height=34.8472213745117)
session.viewports['Viewport: 2'].setValues(origin=(87.1197891235352, 
    115.710639953613), width=87.1197891235352, height=34.8472213745117)
session.viewports['Viewport: 3'].setValues(origin=(174.23957824707, 
    115.710639953613), width=87.1197891235352, height=34.8472213745117)
session.viewports['Viewport: 4'].setValues(origin=(0.0, 80.8634262084961), 
    width=87.1197891235352, height=34.8472213745117)
session.viewports['Viewport: 5'].setValues(origin=(87.1197891235352, 
    80.8634262084961), width=87.1197891235352, height=34.8472213745117)
session.viewports['Viewport: 6'].setValues(origin=(174.23957824707, 
    80.8634262084961), width=87.1197891235352, height=34.8472213745117)
session.viewports['Viewport: 7'].setValues(origin=(0.0, 46.0162048339844), 
    width=87.1197891235352, height=34.8472213745117)
session.viewports['Viewport: 8'].setValues(origin=(87.1197891235352, 
    46.0162048339844), width=87.1197891235352, height=34.8472213745117)
session.viewports['Viewport: 9'].setValues(origin=(174.23957824707, 
    46.0162048339844), width=87.1197891235352, height=34.8472213745117)
session.viewports['Viewport: 8'].makeCurrent()
del session.viewports['Viewport: 9']
session.viewports['Viewport: 1'].setValues(origin=(0.0, 124.42244720459), 
    width=130.791656494141, height=26.1354160308838)
#* RangeError: height must be a Float in the range: 30 <= height <= 3350
session.viewports['Viewport: 2'].setValues(origin=(130.791656494141, 
    124.42244720459), width=130.791656494141, height=26.1354160308838)
#* RangeError: height must be a Float in the range: 30 <= height <= 3350
session.viewports['Viewport: 3'].setValues(origin=(0.0, 98.2870407104492), 
    width=130.791656494141, height=26.1354160308838)
#* RangeError: height must be a Float in the range: 30 <= height <= 3350
session.viewports['Viewport: 4'].setValues(origin=(130.791656494141, 
    98.2870407104492), width=130.791656494141, height=26.1354160308838)
#* RangeError: height must be a Float in the range: 30 <= height <= 3350
session.viewports['Viewport: 5'].setValues(origin=(0.0, 72.1516189575195), 
    width=130.791656494141, height=26.1354160308838)
#* RangeError: height must be a Float in the range: 30 <= height <= 3350
session.viewports['Viewport: 6'].setValues(origin=(130.791656494141, 
    72.1516189575195), width=130.791656494141, height=26.1354160308838)
#* RangeError: height must be a Float in the range: 30 <= height <= 3350
session.viewports['Viewport: 7'].setValues(width=130.791656494141, 
    height=26.1354160308838)
#* RangeError: height must be a Float in the range: 30 <= height <= 3350
session.viewports['Viewport: 8'].setValues(origin=(130.791656494141, 
    46.0162048339844), width=130.791656494141, height=26.1354160308838)
#* RangeError: height must be a Float in the range: 30 <= height <= 3350
session.viewports['Viewport: 1'].setValues(origin=(0.0, 124.42244720459), 
    width=130.791656494141, height=26.1354160308838)
#* RangeError: height must be a Float in the range: 30 <= height <= 3350
session.viewports['Viewport: 2'].setValues(origin=(130.791656494141, 
    124.42244720459), width=130.791656494141, height=26.1354160308838)
#* RangeError: height must be a Float in the range: 30 <= height <= 3350
session.viewports['Viewport: 3'].setValues(origin=(0.0, 98.2870407104492), 
    width=130.791656494141, height=26.1354160308838)
#* RangeError: height must be a Float in the range: 30 <= height <= 3350
session.viewports['Viewport: 4'].setValues(origin=(130.791656494141, 
    98.2870407104492), width=130.791656494141, height=26.1354160308838)
#* RangeError: height must be a Float in the range: 30 <= height <= 3350
session.viewports['Viewport: 5'].setValues(origin=(0.0, 72.1516189575195), 
    width=130.791656494141, height=26.1354160308838)
#* RangeError: height must be a Float in the range: 30 <= height <= 3350
session.viewports['Viewport: 6'].setValues(origin=(130.791656494141, 
    72.1516189575195), width=130.791656494141, height=26.1354160308838)
#* RangeError: height must be a Float in the range: 30 <= height <= 3350
session.viewports['Viewport: 7'].setValues(width=130.791656494141, 
    height=26.1354160308838)
#* RangeError: height must be a Float in the range: 30 <= height <= 3350
session.viewports['Viewport: 8'].setValues(origin=(130.791656494141, 
    46.0162048339844), width=130.791656494141, height=26.1354160308838)
#* RangeError: height must be a Float in the range: 30 <= height <= 3350
session.viewports['Viewport: 1'].setValues(origin=(0.0, 98.2870330810547), 
    width=65.3958282470703, height=52.2708320617676)
session.viewports['Viewport: 2'].setValues(origin=(65.3958282470703, 
    98.2870330810547), width=65.3958282470703, height=52.2708320617676)
session.viewports['Viewport: 3'].setValues(origin=(130.791656494141, 
    98.2870330810547), width=65.3958282470703, height=52.2708320617676)
session.viewports['Viewport: 4'].setValues(origin=(196.1875, 98.2870330810547), 
    width=65.3958282470703, height=52.2708320617676)
session.viewports['Viewport: 5'].setValues(origin=(0.0, 46.0162010192871), 
    width=65.3958282470703, height=52.2708320617676)
session.viewports['Viewport: 6'].setValues(origin=(65.3958282470703, 
    46.0162010192871), width=65.3958282470703, height=52.2708320617676)
session.viewports['Viewport: 7'].setValues(origin=(130.791656494141, 
    46.0162010192871), width=65.3958282470703, height=52.2708320617676)
session.viewports['Viewport: 8'].setValues(origin=(196.1875, 46.0162010192871), 
    width=65.3958282470703, height=52.2708320617676)
session.viewports['Viewport: 8'].viewportAnnotationOptions.setValues(triad=OFF, 
    legend=OFF, title=OFF, state=OFF, annotations=OFF, compass=OFF)
session.viewports['Viewport: 7'].makeCurrent()
session.viewports['Viewport: 7'].viewportAnnotationOptions.setValues(triad=OFF, 
    legend=OFF, title=OFF, state=OFF, annotations=OFF, compass=OFF)
session.viewports['Viewport: 6'].makeCurrent()
session.viewports['Viewport: 6'].viewportAnnotationOptions.setValues(triad=OFF, 
    legend=OFF, title=OFF, state=OFF, annotations=OFF, compass=OFF)
session.viewports['Viewport: 5'].makeCurrent()
session.viewports['Viewport: 5'].viewportAnnotationOptions.setValues(triad=OFF, 
    legend=OFF, title=OFF, state=OFF, annotations=OFF, compass=OFF)
session.viewports['Viewport: 4'].makeCurrent()
session.viewports['Viewport: 4'].viewportAnnotationOptions.setValues(triad=OFF, 
    legend=OFF, title=OFF, state=OFF, annotations=OFF, compass=OFF)
session.viewports['Viewport: 3'].makeCurrent()
session.viewports['Viewport: 3'].viewportAnnotationOptions.setValues(triad=OFF, 
    legend=OFF, title=OFF, state=OFF, annotations=OFF, compass=OFF)
session.viewports['Viewport: 2'].makeCurrent()
session.viewports['Viewport: 2'].viewportAnnotationOptions.setValues(triad=OFF, 
    legend=OFF, title=OFF, state=OFF, annotations=OFF, compass=OFF)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].viewportAnnotationOptions.setValues(triad=OFF, 
    legend=OFF, title=OFF, state=OFF, annotations=OFF, compass=OFF)
session. linkedViewportCommands.setValues(linkViewports=True)
session.viewports['Viewport: 1'].view.setValues(session.views['Back'])
session.viewports['Viewport: 1'].view.setValues(session.views['Front'])
session.viewports['Viewport: 1'].view.setValues(session.views['Front'])
session.viewports['Viewport: 1'].view.setValues(width=0.340769, 
    height=0.245542, viewOffsetX=-0.000967582, viewOffsetY=0.00752763)
session.viewports['Viewport: 2'].view.setValues(width=0.341797, 
    height=0.246283, viewOffsetX=-0.000970501, viewOffsetY=0.00755034)
session.viewports['Viewport: 3'].view.setValues(width=0.341797, 
    height=0.246283, viewOffsetX=-0.000970501, viewOffsetY=0.00755034)
session.viewports['Viewport: 4'].view.setValues(width=0.341797, 
    height=0.246283, viewOffsetX=-0.000970501, viewOffsetY=0.00755034)
session.viewports['Viewport: 5'].view.setValues(width=0.341797, 
    height=0.246283, viewOffsetX=-0.000970501, viewOffsetY=0.00755034)
session.viewports['Viewport: 6'].view.setValues(width=0.341797, 
    height=0.246283, viewOffsetX=-0.000970501, viewOffsetY=0.00755034)
session.viewports['Viewport: 7'].view.setValues(width=0.341797, 
    height=0.246283, viewOffsetX=-0.000970501, viewOffsetY=0.00755034)
session.viewports['Viewport: 8'].view.setValues(width=0.341797, 
    height=0.246283, viewOffsetX=-0.000970501, viewOffsetY=0.00755034)
session.viewports['Viewport: 1'].view.setValues(width=0.190679, 
    height=0.137395, viewOffsetX=-0.00111739, viewOffsetY=0.00869311)
session.viewports['Viewport: 2'].view.setValues(width=0.191378, 
    height=0.137898, viewOffsetX=-0.00112148, viewOffsetY=0.00872497)
session.viewports['Viewport: 3'].view.setValues(width=0.191378, 
    height=0.137898, viewOffsetX=-0.00112148, viewOffsetY=0.00872497)
session.viewports['Viewport: 4'].view.setValues(width=0.191378, 
    height=0.137898, viewOffsetX=-0.00112148, viewOffsetY=0.00872497)
session.viewports['Viewport: 5'].view.setValues(width=0.191378, 
    height=0.137898, viewOffsetX=-0.00112148, viewOffsetY=0.00872497)
session.viewports['Viewport: 6'].view.setValues(width=0.191378, 
    height=0.137898, viewOffsetX=-0.00112148, viewOffsetY=0.00872497)
session.viewports['Viewport: 7'].view.setValues(width=0.191378, 
    height=0.137898, viewOffsetX=-0.00112148, viewOffsetY=0.00872497)
session.viewports['Viewport: 8'].view.setValues(width=0.191378, 
    height=0.137898, viewOffsetX=-0.00112148, viewOffsetY=0.00872497)
session.viewports['Viewport: 1'].view.setValues(width=0.234184, 
    height=0.168742, viewOffsetX=-0.00212487, viewOffsetY=0.0165312)
session.viewports['Viewport: 2'].view.setValues(width=0.234955, 
    height=0.169297, viewOffsetX=-0.00213186, viewOffsetY=0.0165855)
session.viewports['Viewport: 3'].view.setValues(width=0.234955, 
    height=0.169297, viewOffsetX=-0.00213186, viewOffsetY=0.0165855)
session.viewports['Viewport: 4'].view.setValues(width=0.234955, 
    height=0.169297, viewOffsetX=-0.00213186, viewOffsetY=0.0165855)
session.viewports['Viewport: 5'].view.setValues(width=0.234955, 
    height=0.169297, viewOffsetX=-0.00213186, viewOffsetY=0.0165855)
session.viewports['Viewport: 6'].view.setValues(width=0.234955, 
    height=0.169297, viewOffsetX=-0.00213186, viewOffsetY=0.0165855)
session.viewports['Viewport: 7'].view.setValues(width=0.234955, 
    height=0.169297, viewOffsetX=-0.00213186, viewOffsetY=0.0165855)
session.viewports['Viewport: 8'].view.setValues(width=0.234955, 
    height=0.169297, viewOffsetX=-0.00213186, viewOffsetY=0.0165855)
session.viewports['Viewport: 1'].view.setValues(width=0.205162, height=0.14783, 
    viewOffsetX=-0.0025629, viewOffsetY=0.0199389)
session.viewports['Viewport: 2'].view.setValues(width=0.205864, 
    height=0.148336, viewOffsetX=-0.00257166, viewOffsetY=0.0200071)
session.viewports['Viewport: 3'].view.setValues(width=0.205864, 
    height=0.148336, viewOffsetX=-0.00257166, viewOffsetY=0.0200071)
session.viewports['Viewport: 4'].view.setValues(width=0.205864, 
    height=0.148336, viewOffsetX=-0.00257166, viewOffsetY=0.0200071)
session.viewports['Viewport: 5'].view.setValues(width=0.205864, 
    height=0.148336, viewOffsetX=-0.00257166, viewOffsetY=0.0200071)
session.viewports['Viewport: 6'].view.setValues(width=0.205864, 
    height=0.148336, viewOffsetX=-0.00257166, viewOffsetY=0.0200071)
session.viewports['Viewport: 7'].view.setValues(width=0.205864, 
    height=0.148336, viewOffsetX=-0.00257166, viewOffsetY=0.0200071)
session.viewports['Viewport: 8'].view.setValues(width=0.205864, 
    height=0.148336, viewOffsetX=-0.00257166, viewOffsetY=0.0200071)
session.viewports['Viewport: 1'].view.setValues(width=0.20224, height=0.145725, 
    viewOffsetX=-0.0032619, viewOffsetY=0.0253771)
session.viewports['Viewport: 2'].view.setValues(width=0.202921, 
    height=0.146215, viewOffsetX=-0.00327287, viewOffsetY=0.0254625)
session.viewports['Viewport: 3'].view.setValues(width=0.202921, 
    height=0.146215, viewOffsetX=-0.00327287, viewOffsetY=0.0254625)
session.viewports['Viewport: 4'].view.setValues(width=0.202921, 
    height=0.146215, viewOffsetX=-0.00327287, viewOffsetY=0.0254625)
session.viewports['Viewport: 5'].view.setValues(width=0.202921, 
    height=0.146215, viewOffsetX=-0.00327287, viewOffsetY=0.0254625)
session.viewports['Viewport: 6'].view.setValues(width=0.202921, 
    height=0.146215, viewOffsetX=-0.00327287, viewOffsetY=0.0254625)
session.viewports['Viewport: 7'].view.setValues(width=0.202921, 
    height=0.146215, viewOffsetX=-0.00327287, viewOffsetY=0.0254625)
session.viewports['Viewport: 8'].view.setValues(width=0.202921, 
    height=0.146215, viewOffsetX=-0.00327287, viewOffsetY=0.0254625)
session.viewports['Viewport: 1'].view.setValues(width=0.190994, 
    height=0.137622, viewOffsetX=-0.00381945, viewOffsetY=0.0297148)
session.viewports['Viewport: 2'].view.setValues(width=0.19164, height=0.138087, 
    viewOffsetX=-0.00383237, viewOffsetY=0.0298152)
session.viewports['Viewport: 3'].view.setValues(width=0.19164, height=0.138087, 
    viewOffsetX=-0.00383237, viewOffsetY=0.0298152)
session.viewports['Viewport: 4'].view.setValues(width=0.19164, height=0.138087, 
    viewOffsetX=-0.00383237, viewOffsetY=0.0298152)
session.viewports['Viewport: 5'].view.setValues(width=0.19164, height=0.138087, 
    viewOffsetX=-0.00383237, viewOffsetY=0.0298152)
session.viewports['Viewport: 6'].view.setValues(width=0.19164, height=0.138087, 
    viewOffsetX=-0.00383237, viewOffsetY=0.0298152)
session.viewports['Viewport: 7'].view.setValues(width=0.19164, height=0.138087, 
    viewOffsetX=-0.00383237, viewOffsetY=0.0298152)
session.viewports['Viewport: 8'].view.setValues(width=0.19164, height=0.138087, 
    viewOffsetX=-0.00383237, viewOffsetY=0.0298152)
session.viewports['Viewport: 1'].view.setValues(width=0.182749, height=0.13168, 
    viewOffsetX=-0.00378406, viewOffsetY=0.03329)
session.viewports['Viewport: 2'].view.setValues(width=0.183366, 
    height=0.132125, viewOffsetX=-0.00379682, viewOffsetY=0.0334023)
session.viewports['Viewport: 3'].view.setValues(width=0.183366, 
    height=0.132125, viewOffsetX=-0.00379682, viewOffsetY=0.0334023)
session.viewports['Viewport: 4'].view.setValues(width=0.183366, 
    height=0.132125, viewOffsetX=-0.00379682, viewOffsetY=0.0334023)
session.viewports['Viewport: 5'].view.setValues(width=0.183366, 
    height=0.132125, viewOffsetX=-0.00379682, viewOffsetY=0.0334023)
session.viewports['Viewport: 6'].view.setValues(width=0.183366, 
    height=0.132125, viewOffsetX=-0.00379682, viewOffsetY=0.0334023)
session.viewports['Viewport: 7'].view.setValues(width=0.183366, 
    height=0.132125, viewOffsetX=-0.00379682, viewOffsetY=0.0334023)
session.viewports['Viewport: 8'].view.setValues(width=0.183366, 
    height=0.132125, viewOffsetX=-0.00379682, viewOffsetY=0.0334023)
session.viewports['Viewport: 1'].view.setValues(width=0.173999, 
    height=0.125376, viewOffsetX=-0.00357595, viewOffsetY=0.0364591)
session.viewports['Viewport: 2'].view.setValues(width=0.174587, 
    height=0.125799, viewOffsetX=-0.00358801, viewOffsetY=0.0365822)
session.viewports['Viewport: 3'].view.setValues(width=0.174587, 
    height=0.125799, viewOffsetX=-0.00358801, viewOffsetY=0.0365822)
session.viewports['Viewport: 4'].view.setValues(width=0.174587, 
    height=0.125799, viewOffsetX=-0.00358801, viewOffsetY=0.0365822)
session.viewports['Viewport: 5'].view.setValues(width=0.174587, 
    height=0.125799, viewOffsetX=-0.00358801, viewOffsetY=0.0365822)
session.viewports['Viewport: 6'].view.setValues(width=0.174587, 
    height=0.125799, viewOffsetX=-0.00358801, viewOffsetY=0.0365822)
session.viewports['Viewport: 7'].view.setValues(width=0.174587, 
    height=0.125799, viewOffsetX=-0.00358801, viewOffsetY=0.0365822)
session.viewports['Viewport: 8'].view.setValues(width=0.174587, 
    height=0.125799, viewOffsetX=-0.00358801, viewOffsetY=0.0365822)
session.viewports['Viewport: 1'].view.setValues(width=0.165769, 
    height=0.119446, viewOffsetX=-0.0033795, viewOffsetY=0.0395244)
session.viewports['Viewport: 2'].view.setValues(width=0.166328, 
    height=0.119848, viewOffsetX=-0.0033909, viewOffsetY=0.0396577)
session.viewports['Viewport: 3'].view.setValues(width=0.166328, 
    height=0.119848, viewOffsetX=-0.0033909, viewOffsetY=0.0396577)
session.viewports['Viewport: 4'].view.setValues(width=0.166328, 
    height=0.119848, viewOffsetX=-0.0033909, viewOffsetY=0.0396577)
session.viewports['Viewport: 5'].view.setValues(width=0.166328, 
    height=0.119848, viewOffsetX=-0.0033909, viewOffsetY=0.0396577)
session.viewports['Viewport: 6'].view.setValues(width=0.166328, 
    height=0.119848, viewOffsetX=-0.0033909, viewOffsetY=0.0396577)
session.viewports['Viewport: 7'].view.setValues(width=0.166328, 
    height=0.119848, viewOffsetX=-0.0033909, viewOffsetY=0.0396577)
session.viewports['Viewport: 8'].view.setValues(width=0.166328, 
    height=0.119848, viewOffsetX=-0.0033909, viewOffsetY=0.0396577)
session.viewports['Viewport: 1'].view.setValues(width=0.157777, 
    height=0.113687, viewOffsetX=-0.00315308, viewOffsetY=0.0424687)
session.viewports['Viewport: 2'].view.setValues(width=0.158309, height=0.11407, 
    viewOffsetX=-0.00316371, viewOffsetY=0.0426118)
session.viewports['Viewport: 3'].view.setValues(width=0.158309, height=0.11407, 
    viewOffsetX=-0.00316371, viewOffsetY=0.0426118)
session.viewports['Viewport: 4'].view.setValues(width=0.158309, height=0.11407, 
    viewOffsetX=-0.00316371, viewOffsetY=0.0426118)
session.viewports['Viewport: 5'].view.setValues(width=0.158309, height=0.11407, 
    viewOffsetX=-0.00316371, viewOffsetY=0.0426118)
session.viewports['Viewport: 6'].view.setValues(width=0.158309, height=0.11407, 
    viewOffsetX=-0.00316371, viewOffsetY=0.0426118)
session.viewports['Viewport: 7'].view.setValues(width=0.158309, height=0.11407, 
    viewOffsetX=-0.00316371, viewOffsetY=0.0426118)
session.viewports['Viewport: 8'].view.setValues(width=0.158309, height=0.11407, 
    viewOffsetX=-0.00316371, viewOffsetY=0.0426118)
session.viewports['Viewport: 1'].view.setValues(width=0.150094, 
    height=0.108151, viewOffsetX=-0.00256026, viewOffsetY=0.0430304)
session.viewports['Viewport: 2'].view.setValues(width=0.1506, height=0.108515, 
    viewOffsetX=-0.00256889, viewOffsetY=0.0431754)
session.viewports['Viewport: 3'].view.setValues(width=0.1506, height=0.108515, 
    viewOffsetX=-0.00256889, viewOffsetY=0.0431754)
session.viewports['Viewport: 4'].view.setValues(width=0.1506, height=0.108515, 
    viewOffsetX=-0.00256889, viewOffsetY=0.0431754)
session.viewports['Viewport: 5'].view.setValues(width=0.1506, height=0.108515, 
    viewOffsetX=-0.00256889, viewOffsetY=0.0431754)
session.viewports['Viewport: 6'].view.setValues(width=0.1506, height=0.108515, 
    viewOffsetX=-0.00256889, viewOffsetY=0.0431754)
session.viewports['Viewport: 7'].view.setValues(width=0.1506, height=0.108515, 
    viewOffsetX=-0.00256889, viewOffsetY=0.0431754)
session.viewports['Viewport: 8'].view.setValues(width=0.1506, height=0.108515, 
    viewOffsetX=-0.00256889, viewOffsetY=0.0431754)
session.viewports['Viewport: 1'].view.setValues(width=0.1427, height=0.102823, 
    viewOffsetX=-0.001925, viewOffsetY=0.0434088)
session.viewports['Viewport: 2'].view.setValues(width=0.143181, 
    height=0.103169, viewOffsetX=-0.00193149, viewOffsetY=0.043555)
session.viewports['Viewport: 3'].view.setValues(width=0.143181, 
    height=0.103169, viewOffsetX=-0.00193149, viewOffsetY=0.043555)
session.viewports['Viewport: 4'].view.setValues(width=0.143181, 
    height=0.103169, viewOffsetX=-0.00193149, viewOffsetY=0.043555)
session.viewports['Viewport: 5'].view.setValues(width=0.143181, 
    height=0.103169, viewOffsetX=-0.00193149, viewOffsetY=0.043555)
session.viewports['Viewport: 6'].view.setValues(width=0.143181, 
    height=0.103169, viewOffsetX=-0.00193149, viewOffsetY=0.043555)
session.viewports['Viewport: 7'].view.setValues(width=0.143181, 
    height=0.103169, viewOffsetX=-0.00193149, viewOffsetY=0.043555)
session.viewports['Viewport: 8'].view.setValues(width=0.143181, 
    height=0.103169, viewOffsetX=-0.00193149, viewOffsetY=0.043555)
session.viewports['Viewport: 1'].view.setValues(width=0.135596, 
    height=0.097704, viewOffsetX=-0.0011913, viewOffsetY=0.0431893)
session.viewports['Viewport: 2'].view.setValues(width=0.136052, 
    height=0.098033, viewOffsetX=-0.00119532, viewOffsetY=0.0433348)
session.viewports['Viewport: 3'].view.setValues(width=0.136052, 
    height=0.098033, viewOffsetX=-0.00119532, viewOffsetY=0.0433348)
session.viewports['Viewport: 4'].view.setValues(width=0.136052, 
    height=0.098033, viewOffsetX=-0.00119532, viewOffsetY=0.0433348)
session.viewports['Viewport: 5'].view.setValues(width=0.136052, 
    height=0.098033, viewOffsetX=-0.00119532, viewOffsetY=0.0433348)
session.viewports['Viewport: 6'].view.setValues(width=0.136052, 
    height=0.098033, viewOffsetX=-0.00119532, viewOffsetY=0.0433348)
session.viewports['Viewport: 7'].view.setValues(width=0.136052, 
    height=0.098033, viewOffsetX=-0.00119532, viewOffsetY=0.0433348)
session.viewports['Viewport: 8'].view.setValues(width=0.136052, 
    height=0.098033, viewOffsetX=-0.00119532, viewOffsetY=0.0433348)
session.viewports['Viewport: 1'].view.setValues(width=0.155678, 
    height=0.112174, viewOffsetX=-0.0025, viewOffsetY=0.0414141)
session.viewports['Viewport: 2'].view.setValues(width=0.156202, 
    height=0.112552, viewOffsetX=-0.00250842, viewOffsetY=0.0415536)
session.viewports['Viewport: 3'].view.setValues(width=0.156202, 
    height=0.112552, viewOffsetX=-0.00250842, viewOffsetY=0.0415536)
session.viewports['Viewport: 4'].view.setValues(width=0.156202, 
    height=0.112552, viewOffsetX=-0.00250842, viewOffsetY=0.0415536)
session.viewports['Viewport: 5'].view.setValues(width=0.156202, 
    height=0.112552, viewOffsetX=-0.00250842, viewOffsetY=0.0415536)
session.viewports['Viewport: 6'].view.setValues(width=0.156202, 
    height=0.112552, viewOffsetX=-0.00250842, viewOffsetY=0.0415536)
session.viewports['Viewport: 7'].view.setValues(width=0.156202, 
    height=0.112552, viewOffsetX=-0.00250842, viewOffsetY=0.0415536)
session.viewports['Viewport: 8'].view.setValues(width=0.156202, 
    height=0.112552, viewOffsetX=-0.00250842, viewOffsetY=0.0415536)
session.viewports['Viewport: 1'].view.setValues(width=0.160828, 
    height=0.115885, viewOffsetX=-0.00309738, viewOffsetY=0.0387956)
session.viewports['Viewport: 2'].view.setValues(width=0.16137, height=0.116276, 
    viewOffsetX=-0.00310782, viewOffsetY=0.0389264)
session.viewports['Viewport: 3'].view.setValues(width=0.16137, height=0.116276, 
    viewOffsetX=-0.00310782, viewOffsetY=0.0389264)
session.viewports['Viewport: 4'].view.setValues(width=0.16137, height=0.116276, 
    viewOffsetX=-0.00310782, viewOffsetY=0.0389264)
session.viewports['Viewport: 5'].view.setValues(width=0.16137, height=0.116276, 
    viewOffsetX=-0.00310782, viewOffsetY=0.0389264)
session.viewports['Viewport: 6'].view.setValues(width=0.16137, height=0.116276, 
    viewOffsetX=-0.00310782, viewOffsetY=0.0389264)
session.viewports['Viewport: 7'].view.setValues(width=0.16137, height=0.116276, 
    viewOffsetX=-0.00310782, viewOffsetY=0.0389264)
session.viewports['Viewport: 8'].view.setValues(width=0.16137, height=0.116276, 
    viewOffsetX=-0.00310782, viewOffsetY=0.0389264)
session.viewports['Viewport: 1'].view.setValues(width=0.169787, 
    height=0.122341, viewOffsetX=-0.00367192, viewOffsetY=0.0363841)
session.viewports['Viewport: 2'].view.setValues(width=0.17036, height=0.122753, 
    viewOffsetX=-0.0036843, viewOffsetY=0.0365068)
session.viewports['Viewport: 3'].view.setValues(width=0.17036, height=0.122753, 
    viewOffsetX=-0.0036843, viewOffsetY=0.0365068)
session.viewports['Viewport: 4'].view.setValues(width=0.17036, height=0.122753, 
    viewOffsetX=-0.0036843, viewOffsetY=0.0365068)
session.viewports['Viewport: 5'].view.setValues(width=0.17036, height=0.122753, 
    viewOffsetX=-0.0036843, viewOffsetY=0.0365068)
session.viewports['Viewport: 6'].view.setValues(width=0.17036, height=0.122753, 
    viewOffsetX=-0.0036843, viewOffsetY=0.0365068)
session.viewports['Viewport: 7'].view.setValues(width=0.17036, height=0.122753, 
    viewOffsetX=-0.0036843, viewOffsetY=0.0365068)
session.viewports['Viewport: 8'].view.setValues(width=0.17036, height=0.122753, 
    viewOffsetX=-0.0036843, viewOffsetY=0.0365068)
session.viewports['Viewport: 1'].view.setValues(width=0.178208, 
    height=0.128408, viewOffsetX=-0.00409843, viewOffsetY=0.0331077)
session.viewports['Viewport: 2'].view.setValues(width=0.178809, 
    height=0.128841, viewOffsetX=-0.00411225, viewOffsetY=0.0332194)
session.viewports['Viewport: 3'].view.setValues(width=0.178809, 
    height=0.128841, viewOffsetX=-0.00411225, viewOffsetY=0.0332194)
session.viewports['Viewport: 4'].view.setValues(width=0.178809, 
    height=0.128841, viewOffsetX=-0.00411225, viewOffsetY=0.0332194)
session.viewports['Viewport: 5'].view.setValues(width=0.178809, 
    height=0.128841, viewOffsetX=-0.00411225, viewOffsetY=0.0332194)
session.viewports['Viewport: 6'].view.setValues(width=0.178809, 
    height=0.128841, viewOffsetX=-0.00411225, viewOffsetY=0.0332194)
session.viewports['Viewport: 7'].view.setValues(width=0.178809, 
    height=0.128841, viewOffsetX=-0.00411225, viewOffsetY=0.0332194)
session.viewports['Viewport: 8'].view.setValues(width=0.178809, 
    height=0.128841, viewOffsetX=-0.00411225, viewOffsetY=0.0332194)
session.viewports['Viewport: 1'].view.setValues(width=0.16538, height=0.119165, 
    viewOffsetX=-0.00237448, viewOffsetY=0.0293696)
session.viewports['Viewport: 2'].view.setValues(width=0.165938, 
    height=0.119567, viewOffsetX=-0.00238249, viewOffsetY=0.0294687)
session.viewports['Viewport: 3'].view.setValues(width=0.165938, 
    height=0.119567, viewOffsetX=-0.00238249, viewOffsetY=0.0294687)
session.viewports['Viewport: 4'].view.setValues(width=0.165938, 
    height=0.119567, viewOffsetX=-0.00238249, viewOffsetY=0.0294687)
session.viewports['Viewport: 5'].view.setValues(width=0.165938, 
    height=0.119567, viewOffsetX=-0.00238249, viewOffsetY=0.0294687)
session.viewports['Viewport: 6'].view.setValues(width=0.165938, 
    height=0.119567, viewOffsetX=-0.00238249, viewOffsetY=0.0294687)
session.viewports['Viewport: 7'].view.setValues(width=0.165938, 
    height=0.119567, viewOffsetX=-0.00238249, viewOffsetY=0.0294687)
session.viewports['Viewport: 8'].view.setValues(width=0.165938, 
    height=0.119567, viewOffsetX=-0.00238249, viewOffsetY=0.0294687)
session.viewports['Viewport: 1'].view.setValues(width=0.158515, 
    height=0.114219, viewOffsetX=-0.000818867, viewOffsetY=0.0267689)
session.viewports['Viewport: 2'].view.setValues(width=0.159049, 
    height=0.114604, viewOffsetX=-0.000821626, viewOffsetY=0.0268592)
session.viewports['Viewport: 3'].view.setValues(width=0.159049, 
    height=0.114604, viewOffsetX=-0.000821626, viewOffsetY=0.0268592)
session.viewports['Viewport: 4'].view.setValues(width=0.159049, 
    height=0.114604, viewOffsetX=-0.000821626, viewOffsetY=0.0268592)
session.viewports['Viewport: 5'].view.setValues(width=0.159049, 
    height=0.114604, viewOffsetX=-0.000821626, viewOffsetY=0.0268592)
session.viewports['Viewport: 6'].view.setValues(width=0.159049, 
    height=0.114604, viewOffsetX=-0.000821626, viewOffsetY=0.0268592)
session.viewports['Viewport: 7'].view.setValues(width=0.159049, 
    height=0.114604, viewOffsetX=-0.000821626, viewOffsetY=0.0268592)
session.viewports['Viewport: 8'].view.setValues(width=0.159049, 
    height=0.114604, viewOffsetX=-0.000821626, viewOffsetY=0.0268592)
session.viewports['Viewport: 1'].view.setValues(width=0.150542, 
    height=0.108474, viewOffsetX=0.000694405, viewOffsetY=0.0240267)
session.viewports['Viewport: 2'].view.setValues(width=0.15105, height=0.108839, 
    viewOffsetX=0.000696747, viewOffsetY=0.0241077)
session.viewports['Viewport: 3'].view.setValues(width=0.15105, height=0.108839, 
    viewOffsetX=0.000696747, viewOffsetY=0.0241077)
session.viewports['Viewport: 4'].view.setValues(width=0.15105, height=0.108839, 
    viewOffsetX=0.000696747, viewOffsetY=0.0241077)
session.viewports['Viewport: 5'].view.setValues(width=0.15105, height=0.108839, 
    viewOffsetX=0.000696747, viewOffsetY=0.0241077)
session.viewports['Viewport: 6'].view.setValues(width=0.15105, height=0.108839, 
    viewOffsetX=0.000696747, viewOffsetY=0.0241077)
session.viewports['Viewport: 7'].view.setValues(width=0.15105, height=0.108839, 
    viewOffsetX=0.000696747, viewOffsetY=0.0241077)
session.viewports['Viewport: 8'].view.setValues(width=0.15105, height=0.108839, 
    viewOffsetX=0.000696747, viewOffsetY=0.0241077)
session.viewports['Viewport: 1'].view.setValues(width=0.143189, 
    height=0.103175, viewOffsetX=0.00215004, viewOffsetY=0.0214408)
session.viewports['Viewport: 2'].view.setValues(width=0.143672, 
    height=0.103523, viewOffsetX=0.00215729, viewOffsetY=0.021513)
session.viewports['Viewport: 3'].view.setValues(width=0.143672, 
    height=0.103523, viewOffsetX=0.00215729, viewOffsetY=0.021513)
session.viewports['Viewport: 4'].view.setValues(width=0.143672, 
    height=0.103523, viewOffsetX=0.00215729, viewOffsetY=0.021513)
session.viewports['Viewport: 5'].view.setValues(width=0.143672, 
    height=0.103523, viewOffsetX=0.00215729, viewOffsetY=0.021513)
session.viewports['Viewport: 6'].view.setValues(width=0.143672, 
    height=0.103523, viewOffsetX=0.00215729, viewOffsetY=0.021513)
session.viewports['Viewport: 7'].view.setValues(width=0.143672, 
    height=0.103523, viewOffsetX=0.00215729, viewOffsetY=0.021513)
session.viewports['Viewport: 8'].view.setValues(width=0.143672, 
    height=0.103523, viewOffsetX=0.00215729, viewOffsetY=0.021513)
session.viewports['Viewport: 1'].view.setValues(width=0.136054, 
    height=0.0980339, viewOffsetX=0.00385761, viewOffsetY=0.0207633)
session.viewports['Viewport: 2'].view.setValues(width=0.136512, 
    height=0.0983641, viewOffsetX=0.00387061, viewOffsetY=0.0208333)
session.viewports['Viewport: 3'].view.setValues(width=0.136512, 
    height=0.0983641, viewOffsetX=0.00387061, viewOffsetY=0.0208333)
session.viewports['Viewport: 4'].view.setValues(width=0.136512, 
    height=0.0983641, viewOffsetX=0.00387061, viewOffsetY=0.0208333)
session.viewports['Viewport: 5'].view.setValues(width=0.136512, 
    height=0.0983641, viewOffsetX=0.00387061, viewOffsetY=0.0208333)
session.viewports['Viewport: 6'].view.setValues(width=0.136512, 
    height=0.0983641, viewOffsetX=0.00387061, viewOffsetY=0.0208333)
session.viewports['Viewport: 7'].view.setValues(width=0.136512, 
    height=0.0983641, viewOffsetX=0.00387061, viewOffsetY=0.0208333)
session.viewports['Viewport: 8'].view.setValues(width=0.136512, 
    height=0.0983641, viewOffsetX=0.00387061, viewOffsetY=0.0208333)
session.viewports['Viewport: 1'].view.setValues(width=0.129218, 
    height=0.0931087, viewOffsetX=0.00458745, viewOffsetY=0.0201446)
session.viewports['Viewport: 2'].view.setValues(width=0.129653, 
    height=0.0934222, viewOffsetX=0.00460289, viewOffsetY=0.0202124)
session.viewports['Viewport: 3'].view.setValues(width=0.129653, 
    height=0.0934222, viewOffsetX=0.00460289, viewOffsetY=0.0202124)
session.viewports['Viewport: 4'].view.setValues(width=0.129653, 
    height=0.0934222, viewOffsetX=0.00460289, viewOffsetY=0.0202124)
session.viewports['Viewport: 5'].view.setValues(width=0.129653, 
    height=0.0934222, viewOffsetX=0.00460289, viewOffsetY=0.0202124)
session.viewports['Viewport: 6'].view.setValues(width=0.129653, 
    height=0.0934222, viewOffsetX=0.00460289, viewOffsetY=0.0202124)
session.viewports['Viewport: 7'].view.setValues(width=0.129653, 
    height=0.0934222, viewOffsetX=0.00460289, viewOffsetY=0.0202124)
session.viewports['Viewport: 8'].view.setValues(width=0.129653, 
    height=0.0934222, viewOffsetX=0.00460289, viewOffsetY=0.0202124)
session.viewports['Viewport: 1'].view.setValues(width=0.122661, 
    height=0.0883838, viewOffsetX=0.00525952, viewOffsetY=0.0195508)
session.viewports['Viewport: 2'].view.setValues(width=0.123074, 
    height=0.0886813, viewOffsetX=0.00527722, viewOffsetY=0.0196167)
session.viewports['Viewport: 3'].view.setValues(width=0.123074, 
    height=0.0886813, viewOffsetX=0.00527722, viewOffsetY=0.0196167)
session.viewports['Viewport: 4'].view.setValues(width=0.123074, 
    height=0.0886813, viewOffsetX=0.00527722, viewOffsetY=0.0196167)
session.viewports['Viewport: 5'].view.setValues(width=0.123074, 
    height=0.0886813, viewOffsetX=0.00527722, viewOffsetY=0.0196167)
session.viewports['Viewport: 6'].view.setValues(width=0.123074, 
    height=0.0886813, viewOffsetX=0.00527722, viewOffsetY=0.0196167)
session.viewports['Viewport: 7'].view.setValues(width=0.123074, 
    height=0.0886813, viewOffsetX=0.00527722, viewOffsetY=0.0196167)
session.viewports['Viewport: 8'].view.setValues(width=0.123074, 
    height=0.0886813, viewOffsetX=0.00527722, viewOffsetY=0.0196167)
session.viewports['Viewport: 1'].view.setValues(width=0.11638, 
    height=0.0838578, viewOffsetX=0.00590352, viewOffsetY=0.0189822)
session.viewports['Viewport: 2'].view.setValues(width=0.116771, height=0.08414, 
    viewOffsetX=0.00592338, viewOffsetY=0.0190461)
session.viewports['Viewport: 3'].view.setValues(width=0.116771, height=0.08414, 
    viewOffsetX=0.00592338, viewOffsetY=0.0190461)
session.viewports['Viewport: 4'].view.setValues(width=0.116771, height=0.08414, 
    viewOffsetX=0.00592338, viewOffsetY=0.0190461)
session.viewports['Viewport: 5'].view.setValues(width=0.116771, height=0.08414, 
    viewOffsetX=0.00592338, viewOffsetY=0.0190461)
session.viewports['Viewport: 6'].view.setValues(width=0.116771, height=0.08414, 
    viewOffsetX=0.00592338, viewOffsetY=0.0190461)
session.viewports['Viewport: 7'].view.setValues(width=0.116771, height=0.08414, 
    viewOffsetX=0.00592338, viewOffsetY=0.0190461)
session.viewports['Viewport: 8'].view.setValues(width=0.116771, height=0.08414, 
    viewOffsetX=0.00592338, viewOffsetY=0.0190461)
session.viewports['Viewport: 1'].view.setValues(width=0.110368, 
    height=0.0795259, viewOffsetX=0.00646985, viewOffsetY=0.018413)
session.viewports['Viewport: 2'].view.setValues(width=0.110739, 
    height=0.0797935, viewOffsetX=0.00649162, viewOffsetY=0.018475)
session.viewports['Viewport: 3'].view.setValues(width=0.110739, 
    height=0.0797935, viewOffsetX=0.00649162, viewOffsetY=0.018475)
session.viewports['Viewport: 4'].view.setValues(width=0.110739, 
    height=0.0797935, viewOffsetX=0.00649162, viewOffsetY=0.018475)
session.viewports['Viewport: 5'].view.setValues(width=0.110739, 
    height=0.0797935, viewOffsetX=0.00649162, viewOffsetY=0.018475)
session.viewports['Viewport: 6'].view.setValues(width=0.110739, 
    height=0.0797935, viewOffsetX=0.00649162, viewOffsetY=0.018475)
session.viewports['Viewport: 7'].view.setValues(width=0.110739, 
    height=0.0797935, viewOffsetX=0.00649162, viewOffsetY=0.018475)
session.viewports['Viewport: 8'].view.setValues(width=0.110739, 
    height=0.0797935, viewOffsetX=0.00649162, viewOffsetY=0.018475)
session.viewports['Viewport: 1'].view.setValues(width=0.104619, 
    height=0.0753838, viewOffsetX=0.00629857, viewOffsetY=0.0171814)
session.viewports['Viewport: 2'].view.setValues(width=0.104971, 
    height=0.0756373, viewOffsetX=0.00631975, viewOffsetY=0.0172392)
session.viewports['Viewport: 3'].view.setValues(width=0.104971, 
    height=0.0756373, viewOffsetX=0.00631975, viewOffsetY=0.0172392)
session.viewports['Viewport: 4'].view.setValues(width=0.104971, 
    height=0.0756373, viewOffsetX=0.00631975, viewOffsetY=0.0172392)
session.viewports['Viewport: 5'].view.setValues(width=0.104971, 
    height=0.0756373, viewOffsetX=0.00631975, viewOffsetY=0.0172392)
session.viewports['Viewport: 6'].view.setValues(width=0.104971, 
    height=0.0756373, viewOffsetX=0.00631975, viewOffsetY=0.0172392)
session.viewports['Viewport: 7'].view.setValues(width=0.104971, 
    height=0.0756373, viewOffsetX=0.00631975, viewOffsetY=0.0172392)
session.viewports['Viewport: 8'].view.setValues(width=0.104971, 
    height=0.0756373, viewOffsetX=0.00631975, viewOffsetY=0.0172392)
session.viewports['Viewport: 1'].view.setValues(width=0.0991273, 
    height=0.0714265, viewOffsetX=0.00561705, viewOffsetY=0.0160946)
session.viewports['Viewport: 2'].view.setValues(width=0.0994606, 
    height=0.0716667, viewOffsetX=0.00563594, viewOffsetY=0.0161487)
session.viewports['Viewport: 3'].view.setValues(width=0.0994606, 
    height=0.0716667, viewOffsetX=0.00563594, viewOffsetY=0.0161487)
session.viewports['Viewport: 4'].view.setValues(width=0.0994606, 
    height=0.0716667, viewOffsetX=0.00563594, viewOffsetY=0.0161487)
session.viewports['Viewport: 5'].view.setValues(width=0.0994606, 
    height=0.0716667, viewOffsetX=0.00563594, viewOffsetY=0.0161487)
session.viewports['Viewport: 6'].view.setValues(width=0.0994606, 
    height=0.0716667, viewOffsetX=0.00563594, viewOffsetY=0.0161487)
session.viewports['Viewport: 7'].view.setValues(width=0.0994606, 
    height=0.0716667, viewOffsetX=0.00563594, viewOffsetY=0.0161487)
session.viewports['Viewport: 8'].view.setValues(width=0.0994606, 
    height=0.0716667, viewOffsetX=0.00563594, viewOffsetY=0.0161487)
session.viewports['Viewport: 1'].view.setValues(width=0.0938848, 
    height=0.067649, viewOffsetX=0.00496646, viewOffsetY=0.0150571)
session.viewports['Viewport: 2'].view.setValues(width=0.0942004, 
    height=0.0678764, viewOffsetX=0.00498315, viewOffsetY=0.0151077)
session.viewports['Viewport: 3'].view.setValues(width=0.0942004, 
    height=0.0678764, viewOffsetX=0.00498315, viewOffsetY=0.0151077)
session.viewports['Viewport: 4'].view.setValues(width=0.0942004, 
    height=0.0678764, viewOffsetX=0.00498315, viewOffsetY=0.0151077)
session.viewports['Viewport: 5'].view.setValues(width=0.0942004, 
    height=0.0678764, viewOffsetX=0.00498315, viewOffsetY=0.0151077)
session.viewports['Viewport: 6'].view.setValues(width=0.0942004, 
    height=0.0678764, viewOffsetX=0.00498315, viewOffsetY=0.0151077)
session.viewports['Viewport: 7'].view.setValues(width=0.0942004, 
    height=0.0678764, viewOffsetX=0.00498315, viewOffsetY=0.0151077)
session.viewports['Viewport: 8'].view.setValues(width=0.0942004, 
    height=0.0678764, viewOffsetX=0.00498315, viewOffsetY=0.0151077)
session.viewports['Viewport: 1'].view.setValues(width=0.0888844, 
    height=0.064046, viewOffsetX=0.00434588, viewOffsetY=0.0140675)
session.viewports['Viewport: 2'].view.setValues(width=0.0891831, 
    height=0.0642612, viewOffsetX=0.00436048, viewOffsetY=0.0141148)
session.viewports['Viewport: 3'].view.setValues(width=0.0891831, 
    height=0.0642612, viewOffsetX=0.00436048, viewOffsetY=0.0141148)
session.viewports['Viewport: 4'].view.setValues(width=0.0891831, 
    height=0.0642612, viewOffsetX=0.00436048, viewOffsetY=0.0141148)
session.viewports['Viewport: 5'].view.setValues(width=0.0891831, 
    height=0.0642612, viewOffsetX=0.00436048, viewOffsetY=0.0141148)
session.viewports['Viewport: 6'].view.setValues(width=0.0891831, 
    height=0.0642612, viewOffsetX=0.00436048, viewOffsetY=0.0141148)
session.viewports['Viewport: 7'].view.setValues(width=0.0891831, 
    height=0.0642612, viewOffsetX=0.00436048, viewOffsetY=0.0141148)
session.viewports['Viewport: 8'].view.setValues(width=0.0891831, 
    height=0.0642612, viewOffsetX=0.00436048, viewOffsetY=0.0141148)
session.viewports['Viewport: 1'].view.setValues(width=0.0841186, 
    height=0.060612, viewOffsetX=0.00375438, viewOffsetY=0.0131244)
session.viewports['Viewport: 2'].view.setValues(width=0.0844013, 
    height=0.0608156, viewOffsetX=0.003767, viewOffsetY=0.0131685)
session.viewports['Viewport: 3'].view.setValues(width=0.0844013, 
    height=0.0608156, viewOffsetX=0.003767, viewOffsetY=0.0131685)
session.viewports['Viewport: 4'].view.setValues(width=0.0844013, 
    height=0.0608156, viewOffsetX=0.003767, viewOffsetY=0.0131685)
session.viewports['Viewport: 5'].view.setValues(width=0.0844013, 
    height=0.0608156, viewOffsetX=0.003767, viewOffsetY=0.0131685)
session.viewports['Viewport: 6'].view.setValues(width=0.0844013, 
    height=0.0608156, viewOffsetX=0.003767, viewOffsetY=0.0131685)
session.viewports['Viewport: 7'].view.setValues(width=0.0844013, 
    height=0.0608156, viewOffsetX=0.003767, viewOffsetY=0.0131685)
session.viewports['Viewport: 8'].view.setValues(width=0.0844013, 
    height=0.0608156, viewOffsetX=0.003767, viewOffsetY=0.0131685)
session.viewports['Viewport: 1'].view.setValues(width=0.0795797, 
    height=0.0573414, viewOffsetX=0.00317294, viewOffsetY=0.0122261)
session.viewports['Viewport: 2'].view.setValues(width=0.0798471, 
    height=0.0575341, viewOffsetX=0.0031836, viewOffsetY=0.0122672)
session.viewports['Viewport: 3'].view.setValues(width=0.0798471, 
    height=0.0575341, viewOffsetX=0.0031836, viewOffsetY=0.0122672)
session.viewports['Viewport: 4'].view.setValues(width=0.0798471, 
    height=0.0575341, viewOffsetX=0.0031836, viewOffsetY=0.0122672)
session.viewports['Viewport: 5'].view.setValues(width=0.0798471, 
    height=0.0575341, viewOffsetX=0.0031836, viewOffsetY=0.0122672)
session.viewports['Viewport: 6'].view.setValues(width=0.0798471, 
    height=0.0575341, viewOffsetX=0.0031836, viewOffsetY=0.0122672)
session.viewports['Viewport: 7'].view.setValues(width=0.0798471, 
    height=0.0575341, viewOffsetX=0.0031836, viewOffsetY=0.0122672)
session.viewports['Viewport: 8'].view.setValues(width=0.0798471, 
    height=0.0575341, viewOffsetX=0.0031836, viewOffsetY=0.0122672)
session.viewports['Viewport: 1'].view.setValues(width=0.0752599, 
    height=0.0542288, viewOffsetX=0.00255116, viewOffsetY=0.0114053)
session.viewports['Viewport: 2'].view.setValues(width=0.0755127, 
    height=0.0544109, viewOffsetX=0.00255973, viewOffsetY=0.0114436)
session.viewports['Viewport: 3'].view.setValues(width=0.0755127, 
    height=0.0544109, viewOffsetX=0.00255973, viewOffsetY=0.0114436)
session.viewports['Viewport: 4'].view.setValues(width=0.0755127, 
    height=0.0544109, viewOffsetX=0.00255973, viewOffsetY=0.0114436)
session.viewports['Viewport: 5'].view.setValues(width=0.0755127, 
    height=0.0544109, viewOffsetX=0.00255973, viewOffsetY=0.0114436)
session.viewports['Viewport: 6'].view.setValues(width=0.0755127, 
    height=0.0544109, viewOffsetX=0.00255973, viewOffsetY=0.0114436)
session.viewports['Viewport: 7'].view.setValues(width=0.0755127, 
    height=0.0544109, viewOffsetX=0.00255973, viewOffsetY=0.0114436)
session.viewports['Viewport: 8'].view.setValues(width=0.0755127, 
    height=0.0544109, viewOffsetX=0.00255973, viewOffsetY=0.0114436)
session.viewports['Viewport: 1'].view.setValues(width=0.0711514, 
    height=0.0512683, viewOffsetX=0.00195976, viewOffsetY=0.0106246)
session.viewports['Viewport: 2'].view.setValues(width=0.0713903, 
    height=0.0514405, viewOffsetX=0.00196634, viewOffsetY=0.0106602)
session.viewports['Viewport: 3'].view.setValues(width=0.0713903, 
    height=0.0514405, viewOffsetX=0.00196634, viewOffsetY=0.0106602)
session.viewports['Viewport: 4'].view.setValues(width=0.0713903, 
    height=0.0514405, viewOffsetX=0.00196634, viewOffsetY=0.0106602)
session.viewports['Viewport: 5'].view.setValues(width=0.0713903, 
    height=0.0514405, viewOffsetX=0.00196634, viewOffsetY=0.0106602)
session.viewports['Viewport: 6'].view.setValues(width=0.0713903, 
    height=0.0514405, viewOffsetX=0.00196634, viewOffsetY=0.0106602)
session.viewports['Viewport: 7'].view.setValues(width=0.0713903, 
    height=0.0514405, viewOffsetX=0.00196634, viewOffsetY=0.0106602)
session.viewports['Viewport: 8'].view.setValues(width=0.0713903, 
    height=0.0514405, viewOffsetX=0.00196634, viewOffsetY=0.0106602)
session.viewports['Viewport: 1'].view.setValues(width=0.0672462, 
    height=0.0484545, viewOffsetX=0.0013976, viewOffsetY=0.00988249)
session.viewports['Viewport: 2'].view.setValues(width=0.067472, 
    height=0.0486171, viewOffsetX=0.0014023, viewOffsetY=0.00991568)
session.viewports['Viewport: 3'].view.setValues(width=0.067472, 
    height=0.0486171, viewOffsetX=0.0014023, viewOffsetY=0.00991568)
session.viewports['Viewport: 4'].view.setValues(width=0.067472, 
    height=0.0486171, viewOffsetX=0.0014023, viewOffsetY=0.00991568)
session.viewports['Viewport: 5'].view.setValues(width=0.067472, 
    height=0.0486171, viewOffsetX=0.0014023, viewOffsetY=0.00991568)
session.viewports['Viewport: 6'].view.setValues(width=0.067472, 
    height=0.0486171, viewOffsetX=0.0014023, viewOffsetY=0.00991568)
session.viewports['Viewport: 7'].view.setValues(width=0.067472, 
    height=0.0486171, viewOffsetX=0.0014023, viewOffsetY=0.00991568)
session.viewports['Viewport: 8'].view.setValues(width=0.067472, 
    height=0.0486171, viewOffsetX=0.0014023, viewOffsetY=0.00991568)
session.viewports['Viewport: 1'].view.setValues(width=0.0635366, 
    height=0.0457815, viewOffsetX=0.000863574, viewOffsetY=0.00917757)
session.viewports['Viewport: 2'].view.setValues(width=0.0637498, 
    height=0.0459351, viewOffsetX=0.000866475, viewOffsetY=0.0092084)
session.viewports['Viewport: 3'].view.setValues(width=0.0637498, 
    height=0.0459351, viewOffsetX=0.000866475, viewOffsetY=0.0092084)
session.viewports['Viewport: 4'].view.setValues(width=0.0637498, 
    height=0.0459351, viewOffsetX=0.000866475, viewOffsetY=0.0092084)
session.viewports['Viewport: 5'].view.setValues(width=0.0637498, 
    height=0.0459351, viewOffsetX=0.000866475, viewOffsetY=0.0092084)
session.viewports['Viewport: 6'].view.setValues(width=0.0637498, 
    height=0.0459351, viewOffsetX=0.000866475, viewOffsetY=0.0092084)
session.viewports['Viewport: 7'].view.setValues(width=0.0637498, 
    height=0.0459351, viewOffsetX=0.000866475, viewOffsetY=0.0092084)
session.viewports['Viewport: 8'].view.setValues(width=0.0637498, 
    height=0.0459351, viewOffsetX=0.000866475, viewOffsetY=0.0092084)
session.viewports['Viewport: 1'].view.setValues(width=0.0600147, 
    height=0.0432438, viewOffsetX=0.000356554, viewOffsetY=0.00850833)
session.viewports['Viewport: 2'].view.setValues(width=0.0602161, 
    height=0.0433889, viewOffsetX=0.000357753, viewOffsetY=0.0085369)
session.viewports['Viewport: 3'].view.setValues(width=0.0602161, 
    height=0.0433889, viewOffsetX=0.000357753, viewOffsetY=0.0085369)
session.viewports['Viewport: 4'].view.setValues(width=0.0602161, 
    height=0.0433889, viewOffsetX=0.000357753, viewOffsetY=0.0085369)
session.viewports['Viewport: 5'].view.setValues(width=0.0602161, 
    height=0.0433889, viewOffsetX=0.000357753, viewOffsetY=0.0085369)
session.viewports['Viewport: 6'].view.setValues(width=0.0602161, 
    height=0.0433889, viewOffsetX=0.000357753, viewOffsetY=0.0085369)
session.viewports['Viewport: 7'].view.setValues(width=0.0602161, 
    height=0.0433889, viewOffsetX=0.000357753, viewOffsetY=0.0085369)
session.viewports['Viewport: 8'].view.setValues(width=0.0602161, 
    height=0.0433889, viewOffsetX=0.000357753, viewOffsetY=0.0085369)
session.viewports['Viewport: 1'].view.setValues(width=0.0566729, 
    height=0.0408358, viewOffsetX=-7.30669e-05, viewOffsetY=0.0079375)
session.viewports['Viewport: 2'].view.setValues(width=0.0568631, 
    height=0.0409729, viewOffsetX=-7.33095e-05, viewOffsetY=0.00796416)
session.viewports['Viewport: 3'].view.setValues(width=0.0568631, 
    height=0.0409729, viewOffsetX=-7.33095e-05, viewOffsetY=0.00796416)
session.viewports['Viewport: 4'].view.setValues(width=0.0568631, 
    height=0.0409729, viewOffsetX=-7.33095e-05, viewOffsetY=0.00796416)
session.viewports['Viewport: 5'].view.setValues(width=0.0568631, 
    height=0.0409729, viewOffsetX=-7.33095e-05, viewOffsetY=0.00796416)
session.viewports['Viewport: 6'].view.setValues(width=0.0568631, 
    height=0.0409729, viewOffsetX=-7.33095e-05, viewOffsetY=0.00796416)
session.viewports['Viewport: 7'].view.setValues(width=0.0568631, 
    height=0.0409729, viewOffsetX=-7.33095e-05, viewOffsetY=0.00796416)
session.viewports['Viewport: 8'].view.setValues(width=0.0568631, 
    height=0.0409729, viewOffsetX=-7.33095e-05, viewOffsetY=0.00796416)
session.viewports['Viewport: 1'].view.setValues(width=0.0535036, 
    height=0.0385522, viewOffsetX=-0.000456219, viewOffsetY=0.00739614)
session.viewports['Viewport: 2'].view.setValues(width=0.0536831, 
    height=0.0386815, viewOffsetX=-0.000457746, viewOffsetY=0.00742097)
session.viewports['Viewport: 3'].view.setValues(width=0.0536831, 
    height=0.0386815, viewOffsetX=-0.000457746, viewOffsetY=0.00742097)
session.viewports['Viewport: 4'].view.setValues(width=0.0536831, 
    height=0.0386815, viewOffsetX=-0.000457746, viewOffsetY=0.00742097)
session.viewports['Viewport: 5'].view.setValues(width=0.0536831, 
    height=0.0386815, viewOffsetX=-0.000457746, viewOffsetY=0.00742097)
session.viewports['Viewport: 6'].view.setValues(width=0.0536831, 
    height=0.0386815, viewOffsetX=-0.000457746, viewOffsetY=0.00742097)
session.viewports['Viewport: 7'].view.setValues(width=0.0536831, 
    height=0.0386815, viewOffsetX=-0.000457746, viewOffsetY=0.00742097)
session.viewports['Viewport: 8'].view.setValues(width=0.0536831, 
    height=0.0386815, viewOffsetX=-0.000457746, viewOffsetY=0.00742097)
session.viewports['Viewport: 1'].view.setValues(width=0.0504994, 
    height=0.0363875, viewOffsetX=-0.000807954, viewOffsetY=0.00688297)
session.viewports['Viewport: 2'].view.setValues(width=0.0506688, 
    height=0.0365096, viewOffsetX=-0.000810662, viewOffsetY=0.00690607)
session.viewports['Viewport: 3'].view.setValues(width=0.0506688, 
    height=0.0365096, viewOffsetX=-0.000810662, viewOffsetY=0.00690607)
session.viewports['Viewport: 4'].view.setValues(width=0.0506688, 
    height=0.0365096, viewOffsetX=-0.000810662, viewOffsetY=0.00690607)
session.viewports['Viewport: 5'].view.setValues(width=0.0506688, 
    height=0.0365096, viewOffsetX=-0.000810662, viewOffsetY=0.00690607)
session.viewports['Viewport: 6'].view.setValues(width=0.0506688, 
    height=0.0365096, viewOffsetX=-0.000810662, viewOffsetY=0.00690607)
session.viewports['Viewport: 7'].view.setValues(width=0.0506688, 
    height=0.0365096, viewOffsetX=-0.000810662, viewOffsetY=0.00690607)
session.viewports['Viewport: 8'].view.setValues(width=0.0506688, 
    height=0.0365096, viewOffsetX=-0.000810662, viewOffsetY=0.00690607)
session.viewports['Viewport: 1'].view.setValues(width=0.047653, 
    height=0.0343365, viewOffsetX=-0.00111958, viewOffsetY=0.00637515)
session.viewports['Viewport: 2'].view.setValues(width=0.0478128, 
    height=0.0344517, viewOffsetX=-0.00112333, viewOffsetY=0.00639655)
session.viewports['Viewport: 3'].view.setValues(width=0.0478128, 
    height=0.0344517, viewOffsetX=-0.00112333, viewOffsetY=0.00639655)
session.viewports['Viewport: 4'].view.setValues(width=0.0478128, 
    height=0.0344517, viewOffsetX=-0.00112333, viewOffsetY=0.00639655)
session.viewports['Viewport: 5'].view.setValues(width=0.0478128, 
    height=0.0344517, viewOffsetX=-0.00112333, viewOffsetY=0.00639655)
session.viewports['Viewport: 6'].view.setValues(width=0.0478128, 
    height=0.0344517, viewOffsetX=-0.00112333, viewOffsetY=0.00639655)
session.viewports['Viewport: 7'].view.setValues(width=0.0478128, 
    height=0.0344517, viewOffsetX=-0.00112333, viewOffsetY=0.00639655)
session.viewports['Viewport: 8'].view.setValues(width=0.0478128, 
    height=0.0344517, viewOffsetX=-0.00112333, viewOffsetY=0.00639655)
session.viewports['Viewport: 1'].view.setValues(width=0.0449573, 
    height=0.0323941, viewOffsetX=-0.00141471, viewOffsetY=0.00589423)
session.viewports['Viewport: 2'].view.setValues(width=0.0451081, 
    height=0.0325028, viewOffsetX=-0.00141945, viewOffsetY=0.00591401)
session.viewports['Viewport: 3'].view.setValues(width=0.0451081, 
    height=0.0325028, viewOffsetX=-0.00141945, viewOffsetY=0.00591401)
session.viewports['Viewport: 4'].view.setValues(width=0.0451081, 
    height=0.0325028, viewOffsetX=-0.00141945, viewOffsetY=0.00591401)
session.viewports['Viewport: 5'].view.setValues(width=0.0451081, 
    height=0.0325028, viewOffsetX=-0.00141945, viewOffsetY=0.00591401)
session.viewports['Viewport: 6'].view.setValues(width=0.0451081, 
    height=0.0325028, viewOffsetX=-0.00141945, viewOffsetY=0.00591401)
session.viewports['Viewport: 7'].view.setValues(width=0.0451081, 
    height=0.0325028, viewOffsetX=-0.00141945, viewOffsetY=0.00591401)
session.viewports['Viewport: 8'].view.setValues(width=0.0451081, 
    height=0.0325028, viewOffsetX=-0.00141945, viewOffsetY=0.00591401)
session.viewports['Viewport: 1'].view.setValues(width=0.0424054, 
    height=0.0305554, viewOffsetX=-0.00144366, viewOffsetY=0.0053813)
session.viewports['Viewport: 2'].view.setValues(width=0.0425476, 
    height=0.0306578, viewOffsetX=-0.0014485, viewOffsetY=0.00539937)
session.viewports['Viewport: 3'].view.setValues(width=0.0425476, 
    height=0.0306578, viewOffsetX=-0.0014485, viewOffsetY=0.00539937)
session.viewports['Viewport: 4'].view.setValues(width=0.0425476, 
    height=0.0306578, viewOffsetX=-0.0014485, viewOffsetY=0.00539937)
session.viewports['Viewport: 5'].view.setValues(width=0.0425476, 
    height=0.0306578, viewOffsetX=-0.0014485, viewOffsetY=0.00539937)
session.viewports['Viewport: 6'].view.setValues(width=0.0425476, 
    height=0.0306578, viewOffsetX=-0.0014485, viewOffsetY=0.00539937)
session.viewports['Viewport: 7'].view.setValues(width=0.0425476, 
    height=0.0306578, viewOffsetX=-0.0014485, viewOffsetY=0.00539937)
session.viewports['Viewport: 8'].view.setValues(width=0.0425476, 
    height=0.0306578, viewOffsetX=-0.0014485, viewOffsetY=0.00539937)
session.viewports['Viewport: 1'].view.setValues(width=0.0399906, 
    height=0.0288154, viewOffsetX=-0.00147106, viewOffsetY=0.00489593)
session.viewports['Viewport: 2'].view.setValues(width=0.0401247, 
    height=0.028912, viewOffsetX=-0.00147599, viewOffsetY=0.00491237)
session.viewports['Viewport: 3'].view.setValues(width=0.0401247, 
    height=0.028912, viewOffsetX=-0.00147599, viewOffsetY=0.00491237)
session.viewports['Viewport: 4'].view.setValues(width=0.0401247, 
    height=0.028912, viewOffsetX=-0.00147599, viewOffsetY=0.00491237)
session.viewports['Viewport: 5'].view.setValues(width=0.0401247, 
    height=0.028912, viewOffsetX=-0.00147599, viewOffsetY=0.00491237)
session.viewports['Viewport: 6'].view.setValues(width=0.0401247, 
    height=0.028912, viewOffsetX=-0.00147599, viewOffsetY=0.00491237)
session.viewports['Viewport: 7'].view.setValues(width=0.0401247, 
    height=0.028912, viewOffsetX=-0.00147599, viewOffsetY=0.00491237)
session.viewports['Viewport: 8'].view.setValues(width=0.0401247, 
    height=0.028912, viewOffsetX=-0.00147599, viewOffsetY=0.00491237)
session.viewports['Viewport: 1'].view.setValues(width=0.0377064, 
    height=0.0271695, viewOffsetX=-0.00148842, viewOffsetY=0.0044368)
session.viewports['Viewport: 2'].view.setValues(width=0.0378328, 
    height=0.0272605, viewOffsetX=-0.00149341, viewOffsetY=0.00445169)
session.viewports['Viewport: 3'].view.setValues(width=0.0378328, 
    height=0.0272605, viewOffsetX=-0.00149341, viewOffsetY=0.00445169)
session.viewports['Viewport: 4'].view.setValues(width=0.0378328, 
    height=0.0272605, viewOffsetX=-0.00149341, viewOffsetY=0.00445169)
session.viewports['Viewport: 5'].view.setValues(width=0.0378328, 
    height=0.0272605, viewOffsetX=-0.00149341, viewOffsetY=0.00445169)
session.viewports['Viewport: 6'].view.setValues(width=0.0378328, 
    height=0.0272605, viewOffsetX=-0.00149341, viewOffsetY=0.00445169)
session.viewports['Viewport: 7'].view.setValues(width=0.0378328, 
    height=0.0272605, viewOffsetX=-0.00149341, viewOffsetY=0.00445169)
session.viewports['Viewport: 8'].view.setValues(width=0.0378328, 
    height=0.0272605, viewOffsetX=-0.00149341, viewOffsetY=0.00445169)
session.viewports['Viewport: 1'].view.setValues(width=0.0355464, 
    height=0.0256131, viewOffsetX=-0.00150483, viewOffsetY=0.00400265)
session.viewports['Viewport: 2'].view.setValues(width=0.0356656, 
    height=0.025699, viewOffsetX=-0.00150987, viewOffsetY=0.00401609)
session.viewports['Viewport: 3'].view.setValues(width=0.0356656, 
    height=0.025699, viewOffsetX=-0.00150987, viewOffsetY=0.00401609)
session.viewports['Viewport: 4'].view.setValues(width=0.0356656, 
    height=0.025699, viewOffsetX=-0.00150987, viewOffsetY=0.00401609)
session.viewports['Viewport: 5'].view.setValues(width=0.0356656, 
    height=0.025699, viewOffsetX=-0.00150987, viewOffsetY=0.00401609)
session.viewports['Viewport: 6'].view.setValues(width=0.0356656, 
    height=0.025699, viewOffsetX=-0.00150987, viewOffsetY=0.00401609)
session.viewports['Viewport: 7'].view.setValues(width=0.0356656, 
    height=0.025699, viewOffsetX=-0.00150987, viewOffsetY=0.00401609)
session.viewports['Viewport: 8'].view.setValues(width=0.0356656, 
    height=0.025699, viewOffsetX=-0.00150987, viewOffsetY=0.00401609)
session.viewports['Viewport: 1'].view.setValues(width=0.0335047, 
    height=0.0241419, viewOffsetX=-0.00145185, viewOffsetY=0.0038048)
session.viewports['Viewport: 2'].view.setValues(width=0.033617, 
    height=0.0242228, viewOffsetX=-0.00145672, viewOffsetY=0.00381758)
session.viewports['Viewport: 3'].view.setValues(width=0.033617, 
    height=0.0242228, viewOffsetX=-0.00145672, viewOffsetY=0.00381758)
session.viewports['Viewport: 4'].view.setValues(width=0.033617, 
    height=0.0242228, viewOffsetX=-0.00145672, viewOffsetY=0.00381758)
session.viewports['Viewport: 5'].view.setValues(width=0.033617, 
    height=0.0242228, viewOffsetX=-0.00145672, viewOffsetY=0.00381758)
session.viewports['Viewport: 6'].view.setValues(width=0.033617, 
    height=0.0242228, viewOffsetX=-0.00145672, viewOffsetY=0.00381758)
session.viewports['Viewport: 7'].view.setValues(width=0.033617, 
    height=0.0242228, viewOffsetX=-0.00145672, viewOffsetY=0.00381758)
session.viewports['Viewport: 8'].view.setValues(width=0.033617, 
    height=0.0242228, viewOffsetX=-0.00145672, viewOffsetY=0.00381758)
session.viewports['Viewport: 1'].view.setValues(width=0.0315753, 
    height=0.0227517, viewOffsetX=-0.00140179, viewOffsetY=0.00361785)
session.viewports['Viewport: 2'].view.setValues(width=0.0316811, 
    height=0.0228279, viewOffsetX=-0.00140649, viewOffsetY=0.00362999)
session.viewports['Viewport: 3'].view.setValues(width=0.0316811, 
    height=0.0228279, viewOffsetX=-0.00140649, viewOffsetY=0.00362999)
session.viewports['Viewport: 4'].view.setValues(width=0.0316811, 
    height=0.0228279, viewOffsetX=-0.00140649, viewOffsetY=0.00362999)
session.viewports['Viewport: 5'].view.setValues(width=0.0316811, 
    height=0.0228279, viewOffsetX=-0.00140649, viewOffsetY=0.00362999)
session.viewports['Viewport: 6'].view.setValues(width=0.0316811, 
    height=0.0228279, viewOffsetX=-0.00140649, viewOffsetY=0.00362999)
session.viewports['Viewport: 7'].view.setValues(width=0.0316811, 
    height=0.0228279, viewOffsetX=-0.00140649, viewOffsetY=0.00362999)
session.viewports['Viewport: 8'].view.setValues(width=0.0316811, 
    height=0.0228279, viewOffsetX=-0.00140649, viewOffsetY=0.00362999)
session.viewports['Viewport: 1'].view.setValues(width=0.0297527, 
    height=0.0214384, viewOffsetX=-0.0013545, viewOffsetY=0.00344123)
session.viewports['Viewport: 2'].view.setValues(width=0.0298524, 
    height=0.0215102, viewOffsetX=-0.00135903, viewOffsetY=0.00345278)
session.viewports['Viewport: 3'].view.setValues(width=0.0298524, 
    height=0.0215102, viewOffsetX=-0.00135903, viewOffsetY=0.00345278)
session.viewports['Viewport: 4'].view.setValues(width=0.0298524, 
    height=0.0215102, viewOffsetX=-0.00135903, viewOffsetY=0.00345278)
session.viewports['Viewport: 5'].view.setValues(width=0.0298524, 
    height=0.0215102, viewOffsetX=-0.00135903, viewOffsetY=0.00345278)
session.viewports['Viewport: 6'].view.setValues(width=0.0298524, 
    height=0.0215102, viewOffsetX=-0.00135903, viewOffsetY=0.00345278)
session.viewports['Viewport: 7'].view.setValues(width=0.0298524, 
    height=0.0215102, viewOffsetX=-0.00135903, viewOffsetY=0.00345278)
session.viewports['Viewport: 8'].view.setValues(width=0.0298524, 
    height=0.0215102, viewOffsetX=-0.00135903, viewOffsetY=0.00345278)
session.viewports['Viewport: 1'].view.setValues(width=0.0280313, 
    height=0.0201981, viewOffsetX=-0.00130983, viewOffsetY=0.00327443)
session.viewports['Viewport: 2'].view.setValues(width=0.0281252, 
    height=0.0202657, viewOffsetX=-0.00131422, viewOffsetY=0.00328542)
session.viewports['Viewport: 3'].view.setValues(width=0.0281252, 
    height=0.0202657, viewOffsetX=-0.00131422, viewOffsetY=0.00328542)
session.viewports['Viewport: 4'].view.setValues(width=0.0281252, 
    height=0.0202657, viewOffsetX=-0.00131422, viewOffsetY=0.00328542)
session.viewports['Viewport: 5'].view.setValues(width=0.0281252, 
    height=0.0202657, viewOffsetX=-0.00131422, viewOffsetY=0.00328542)
session.viewports['Viewport: 6'].view.setValues(width=0.0281252, 
    height=0.0202657, viewOffsetX=-0.00131422, viewOffsetY=0.00328542)
session.viewports['Viewport: 7'].view.setValues(width=0.0281252, 
    height=0.0202657, viewOffsetX=-0.00131422, viewOffsetY=0.00328542)
session.viewports['Viewport: 8'].view.setValues(width=0.0281252, 
    height=0.0202657, viewOffsetX=-0.00131422, viewOffsetY=0.00328542)
session.viewports['Viewport: 1'].view.setValues(width=0.0264061, 
    height=0.019027, viewOffsetX=-0.00135763, viewOffsetY=0.00307507)
session.viewports['Viewport: 2'].view.setValues(width=0.0264946, 
    height=0.0190907, viewOffsetX=-0.00136218, viewOffsetY=0.00308539)
session.viewports['Viewport: 3'].view.setValues(width=0.0264946, 
    height=0.0190907, viewOffsetX=-0.00136218, viewOffsetY=0.00308539)
session.viewports['Viewport: 4'].view.setValues(width=0.0264946, 
    height=0.0190907, viewOffsetX=-0.00136218, viewOffsetY=0.00308539)
session.viewports['Viewport: 5'].view.setValues(width=0.0264946, 
    height=0.0190907, viewOffsetX=-0.00136218, viewOffsetY=0.00308539)
session.viewports['Viewport: 6'].view.setValues(width=0.0264946, 
    height=0.0190907, viewOffsetX=-0.00136218, viewOffsetY=0.00308539)
session.viewports['Viewport: 7'].view.setValues(width=0.0264946, 
    height=0.0190907, viewOffsetX=-0.00136218, viewOffsetY=0.00308539)
session.viewports['Viewport: 8'].view.setValues(width=0.0264946, 
    height=0.0190907, viewOffsetX=-0.00136218, viewOffsetY=0.00308539)
session.viewports['Viewport: 1'].view.setValues(width=0.024872, 
    height=0.0179216, viewOffsetX=-0.00140841, viewOffsetY=0.00286435)
session.viewports['Viewport: 2'].view.setValues(width=0.0249553, 
    height=0.0179816, viewOffsetX=-0.00141312, viewOffsetY=0.00287396)
session.viewports['Viewport: 3'].view.setValues(width=0.0249553, 
    height=0.0179816, viewOffsetX=-0.00141312, viewOffsetY=0.00287396)
session.viewports['Viewport: 4'].view.setValues(width=0.0249553, 
    height=0.0179816, viewOffsetX=-0.00141312, viewOffsetY=0.00287396)
session.viewports['Viewport: 5'].view.setValues(width=0.0249553, 
    height=0.0179816, viewOffsetX=-0.00141312, viewOffsetY=0.00287396)
session.viewports['Viewport: 6'].view.setValues(width=0.0249553, 
    height=0.0179816, viewOffsetX=-0.00141312, viewOffsetY=0.00287396)
session.viewports['Viewport: 7'].view.setValues(width=0.0249553, 
    height=0.0179816, viewOffsetX=-0.00141312, viewOffsetY=0.00287396)
session.viewports['Viewport: 8'].view.setValues(width=0.0249553, 
    height=0.0179816, viewOffsetX=-0.00141312, viewOffsetY=0.00287396)
session.viewports['Viewport: 1'].view.setValues(width=0.0234243, 
    height=0.0168785, viewOffsetX=-0.00145632, viewOffsetY=0.00266549)
session.viewports['Viewport: 2'].view.setValues(width=0.0235028, 
    height=0.016935, viewOffsetX=-0.0014612, viewOffsetY=0.00267444)
session.viewports['Viewport: 3'].view.setValues(width=0.0235028, 
    height=0.016935, viewOffsetX=-0.0014612, viewOffsetY=0.00267444)
session.viewports['Viewport: 4'].view.setValues(width=0.0235028, 
    height=0.016935, viewOffsetX=-0.0014612, viewOffsetY=0.00267444)
session.viewports['Viewport: 5'].view.setValues(width=0.0235028, 
    height=0.016935, viewOffsetX=-0.0014612, viewOffsetY=0.00267444)
session.viewports['Viewport: 6'].view.setValues(width=0.0235028, 
    height=0.016935, viewOffsetX=-0.0014612, viewOffsetY=0.00267444)
session.viewports['Viewport: 7'].view.setValues(width=0.0235028, 
    height=0.016935, viewOffsetX=-0.0014612, viewOffsetY=0.00267444)
session.viewports['Viewport: 8'].view.setValues(width=0.0235028, 
    height=0.016935, viewOffsetX=-0.0014612, viewOffsetY=0.00267444)
session.viewports['Viewport: 1'].view.setValues(width=0.0220584, 
    height=0.0158943, viewOffsetX=-0.00150153, viewOffsetY=0.00247787)
session.viewports['Viewport: 2'].view.setValues(width=0.0221323, 
    height=0.0159475, viewOffsetX=-0.00150656, viewOffsetY=0.00248619)
session.viewports['Viewport: 3'].view.setValues(width=0.0221323, 
    height=0.0159475, viewOffsetX=-0.00150656, viewOffsetY=0.00248619)
session.viewports['Viewport: 4'].view.setValues(width=0.0221323, 
    height=0.0159475, viewOffsetX=-0.00150656, viewOffsetY=0.00248619)
session.viewports['Viewport: 5'].view.setValues(width=0.0221323, 
    height=0.0159475, viewOffsetX=-0.00150656, viewOffsetY=0.00248619)
session.viewports['Viewport: 6'].view.setValues(width=0.0221323, 
    height=0.0159475, viewOffsetX=-0.00150656, viewOffsetY=0.00248619)
session.viewports['Viewport: 7'].view.setValues(width=0.0221323, 
    height=0.0159475, viewOffsetX=-0.00150656, viewOffsetY=0.00248619)
session.viewports['Viewport: 8'].view.setValues(width=0.0221323, 
    height=0.0159475, viewOffsetX=-0.00150656, viewOffsetY=0.00248619)
session.viewports['Viewport: 1'].view.setValues(width=0.02077, 
    height=0.0149659, viewOffsetX=-0.00154417, viewOffsetY=0.00229618)
session.viewports['Viewport: 2'].view.setValues(width=0.0208396, 
    height=0.015016, viewOffsetX=-0.00154934, viewOffsetY=0.00230389)
session.viewports['Viewport: 3'].view.setValues(width=0.0208396, 
    height=0.015016, viewOffsetX=-0.00154934, viewOffsetY=0.00230389)
session.viewports['Viewport: 4'].view.setValues(width=0.0208396, 
    height=0.015016, viewOffsetX=-0.00154934, viewOffsetY=0.00230389)
session.viewports['Viewport: 5'].view.setValues(width=0.0208396, 
    height=0.015016, viewOffsetX=-0.00154934, viewOffsetY=0.00230389)
session.viewports['Viewport: 6'].view.setValues(width=0.0208396, 
    height=0.015016, viewOffsetX=-0.00154934, viewOffsetY=0.00230389)
session.viewports['Viewport: 7'].view.setValues(width=0.0208396, 
    height=0.015016, viewOffsetX=-0.00154934, viewOffsetY=0.00230389)
session.viewports['Viewport: 8'].view.setValues(width=0.0208396, 
    height=0.015016, viewOffsetX=-0.00154934, viewOffsetY=0.00230389)
session.viewports['Viewport: 2'].makeCurrent()
session.viewports['Viewport: 2'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM1', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 2'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM2', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 2'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM1', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 2'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 4'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 6'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 8'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 3'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 5'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 7'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 2'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 4'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 6'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 8'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 3'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 5'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 7'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 2'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 4'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 6'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 8'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 3'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 5'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 7'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 2'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 4'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 6'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 8'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 3'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 5'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 7'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 1'].makeCurrent()
o3 = session.odbs['C:/LocalUserData/User-data/nguyenb5/CP1000 plastic (UMAT UMATHT)/CP1000_dense_CSC/CP1000_CHD4_dense_CSC/CHD4_combined_processed.odb']
session.viewports['Viewport: 1'].setValues(displayedObject=o3)
session.viewports['Viewport: 1'].view.setValues(session.views['Front'])
session. linkedViewportCommands.setValues(linkViewports=False)
session.viewports['Viewport: 2'].makeCurrent()
o3 = session.odbs['C:/LocalUserData/User-data/nguyenb5/CP1000 plastic (UMAT UMATHT)/CP1000_dense_CSC/CP1000_NDBR2p5_dense_CSC/NDBR2p5_combined_processed.odb']
session.viewports['Viewport: 2'].setValues(displayedObject=o3)
session.viewports['Viewport: 3'].makeCurrent()
session.viewports['Viewport: 4'].makeCurrent()
o3 = session.odbs['C:/LocalUserData/User-data/nguyenb5/CP1000 plastic (UMAT UMATHT)/CP1000_dense_CSC/CP1000_SH115_dense_CSC/SH115_combined_processed.odb']
session.viewports['Viewport: 4'].setValues(displayedObject=o3)
session.viewports['Viewport: 5'].makeCurrent()
o3 = session.odbs['C:/LocalUserData/User-data/nguyenb5/CP1000 plastic (UMAT UMATHT)/CP1000_dense_CSC/CP1000_CHD4_dense_CSC/CHD4_combined_processed.odb']
session.viewports['Viewport: 5'].setValues(displayedObject=o3)
session.viewports['Viewport: 6'].makeCurrent()
o3 = session.odbs['C:/LocalUserData/User-data/nguyenb5/CP1000 plastic (UMAT UMATHT)/CP1000_dense_CSC/CP1000_NDBR2p5_dense_CSC/NDBR2p5_combined_processed.odb']
session.viewports['Viewport: 6'].setValues(displayedObject=o3)
session.viewports['Viewport: 7'].makeCurrent()
session.viewports['Viewport: 8'].makeCurrent()
o3 = session.odbs['C:/LocalUserData/User-data/nguyenb5/CP1000 plastic (UMAT UMATHT)/CP1000_dense_CSC/CP1000_SH115_dense_CSC/SH115_combined_processed.odb']
session.viewports['Viewport: 8'].setValues(displayedObject=o3)
session.viewports['Viewport: 1'].makeCurrent()
session.linkedViewportCommands.setValues(linkViewports=True)
session.viewports['Viewport: 5'].setValues(applyLinkedCommands=False)
session.viewports['Viewport: 6'].setValues(applyLinkedCommands=False)
session.viewports['Viewport: 7'].setValues(applyLinkedCommands=False)
session.viewports['Viewport: 8'].setValues(applyLinkedCommands=False)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM2', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].odbDisplay.display.setValues(
    plotState=CONTOURS_ON_DEF)
session.viewports['Viewport: 1'].view.setValues(session.views['Back'])
session.viewports['Viewport: 1'].view.setValues(session.views['Front'])
session.viewports['Viewport: 1'].view.setValues(session.views['Back'])
session.viewports['Viewport: 1'].view.setValues(width=0.341242, 
    height=0.245883, viewOffsetX=-0.00011627, viewOffsetY=-3.8656e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.340938, 
    height=0.245664, viewOffsetX=-0.000116167, viewOffsetY=-3.8622e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.341797, 
    height=0.246283, viewOffsetX=-0.00011646, viewOffsetY=-3.87195e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.342119, 
    height=0.246515, viewOffsetX=-0.00011657, viewOffsetY=-3.87552e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.190945, 
    height=0.137586, viewOffsetX=-0.000134273, viewOffsetY=-4.46417e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.190772, 
    height=0.137461, viewOffsetX=-0.000134152, viewOffsetY=-4.46014e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.191378, 
    height=0.137898, viewOffsetX=-0.000134578, viewOffsetY=-4.47429e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.191509, 
    height=0.137992, viewOffsetX=-0.00013467, viewOffsetY=-4.47735e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.23451, height=0.168977, 
    viewOffsetX=-0.000255338, viewOffsetY=-8.48919e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.2343, height=0.168825, 
    viewOffsetX=-0.00025511, viewOffsetY=-8.48162e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.234955, 
    height=0.169297, viewOffsetX=-0.000255822, viewOffsetY=-8.50531e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.235154, 
    height=0.169441, viewOffsetX=-0.000256041, viewOffsetY=-8.51255e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.205448, 
    height=0.148036, viewOffsetX=-0.000307975, viewOffsetY=-0.000102392)
session.viewports['Viewport: 2'].view.setValues(width=0.205263, 
    height=0.147903, viewOffsetX=-0.000307699, viewOffsetY=-0.0001023)
session.viewports['Viewport: 3'].view.setValues(width=0.205864, 
    height=0.148336, viewOffsetX=-0.000308599, viewOffsetY=-0.000102599)
session.viewports['Viewport: 4'].view.setValues(width=0.206026, 
    height=0.148453, viewOffsetX=-0.000308843, viewOffsetY=-0.00010268)
session.viewports['Viewport: 1'].view.setValues(width=0.202522, 
    height=0.145928, viewOffsetX=-0.000391972, viewOffsetY=-0.000130318)
session.viewports['Viewport: 2'].view.setValues(width=0.20234, height=0.145797, 
    viewOffsetX=-0.000391621, viewOffsetY=-0.000130202)
session.viewports['Viewport: 3'].view.setValues(width=0.202921, 
    height=0.146215, viewOffsetX=-0.000392744, viewOffsetY=-0.000130575)
session.viewports['Viewport: 4'].view.setValues(width=0.203087, 
    height=0.146335, viewOffsetX=-0.000393066, viewOffsetY=-0.000130682)
session.viewports['Viewport: 1'].view.setValues(width=0.19126, height=0.137813, 
    viewOffsetX=-0.000458972, viewOffsetY=2.0738e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.191088, 
    height=0.137689, viewOffsetX=-0.00045856, viewOffsetY=2.07191e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.19164, height=0.138087, 
    viewOffsetX=-0.000459884, viewOffsetY=2.07787e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.191795, 
    height=0.138199, viewOffsetX=-0.000460257, viewOffsetY=2.07964e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.183003, 
    height=0.131864, viewOffsetX=-0.000529543, viewOffsetY=0.000166227)
session.viewports['Viewport: 2'].view.setValues(width=0.182839, 
    height=0.131745, viewOffsetX=-0.000529068, viewOffsetY=0.000166077)
session.viewports['Viewport: 3'].view.setValues(width=0.183366, 
    height=0.132125, viewOffsetX=-0.000530591, viewOffsetY=0.000166556)
session.viewports['Viewport: 4'].view.setValues(width=0.183515, 
    height=0.132232, viewOffsetX=-0.000531025, viewOffsetY=0.000166693)
session.viewports['Viewport: 1'].view.setValues(width=0.174241, height=0.12555, 
    viewOffsetX=-0.000595741, viewOffsetY=0.000306541)
session.viewports['Viewport: 2'].view.setValues(width=0.174085, 
    height=0.125437, viewOffsetX=-0.000595206, viewOffsetY=0.000306265)
session.viewports['Viewport: 3'].view.setValues(width=0.174587, 
    height=0.125799, viewOffsetX=-0.000596921, viewOffsetY=0.000307147)
session.viewports['Viewport: 4'].view.setValues(width=0.174729, 
    height=0.125901, viewOffsetX=-0.000597408, viewOffsetY=0.000307398)
session.viewports['Viewport: 1'].view.setValues(width=0.166, height=0.119612, 
    viewOffsetX=-0.000660351, viewOffsetY=0.000442317)
session.viewports['Viewport: 2'].view.setValues(width=0.165851, 
    height=0.119504, viewOffsetX=-0.000659758, viewOffsetY=0.000441919)
session.viewports['Viewport: 3'].view.setValues(width=0.166328, 
    height=0.119848, viewOffsetX=-0.000661657, viewOffsetY=0.000443192)
session.viewports['Viewport: 4'].view.setValues(width=0.166464, 
    height=0.119946, viewOffsetX=-0.000662198, viewOffsetY=0.000443554)
session.viewports['Viewport: 1'].view.setValues(width=0.157996, 
    height=0.113845, viewOffsetX=-0.000722465, viewOffsetY=0.000573151)
session.viewports['Viewport: 2'].view.setValues(width=0.157854, 
    height=0.113742, viewOffsetX=-0.000721816, viewOffsetY=0.000572635)
session.viewports['Viewport: 3'].view.setValues(width=0.158309, height=0.11407, 
    viewOffsetX=-0.000723893, viewOffsetY=0.000574284)
session.viewports['Viewport: 4'].view.setValues(width=0.158438, 
    height=0.114163, viewOffsetX=-0.000724486, viewOffsetY=0.000574753)
session.viewports['Viewport: 1'].view.setValues(width=0.150303, 
    height=0.108301, viewOffsetX=-0.000782366, viewOffsetY=0.00059707)
session.viewports['Viewport: 2'].view.setValues(width=0.150168, 
    height=0.108204, viewOffsetX=-0.000781663, viewOffsetY=0.000596533)
session.viewports['Viewport: 3'].view.setValues(width=0.1506, height=0.108515, 
    viewOffsetX=-0.000783911, viewOffsetY=0.00059825)
session.viewports['Viewport: 4'].view.setValues(width=0.150723, 
    height=0.108604, viewOffsetX=-0.000784554, viewOffsetY=0.00059874)
session.viewports['Viewport: 1'].view.setValues(width=0.142898, 
    height=0.102966, viewOffsetX=-0.000839992, viewOffsetY=0.000620078)
session.viewports['Viewport: 2'].view.setValues(width=0.14277, height=0.102873, 
    viewOffsetX=-0.000839238, viewOffsetY=0.00061952)
session.viewports['Viewport: 3'].view.setValues(width=0.143181, 
    height=0.103169, viewOffsetX=-0.000841651, viewOffsetY=0.000621303)
session.viewports['Viewport: 4'].view.setValues(width=0.143298, 
    height=0.103254, viewOffsetX=-0.000842342, viewOffsetY=0.000621812)
session.viewports['Viewport: 1'].view.setValues(width=0.135784, 
    height=0.0978399, viewOffsetX=-0.000895388, viewOffsetY=0.0006422)
session.viewports['Viewport: 2'].view.setValues(width=0.135662, 
    height=0.097752, viewOffsetX=-0.000894584, viewOffsetY=0.000641622)
session.viewports['Viewport: 3'].view.setValues(width=0.136052, 
    height=0.098033, viewOffsetX=-0.000897155, viewOffsetY=0.000643468)
session.viewports['Viewport: 4'].view.setValues(width=0.136164, 
    height=0.0981134, viewOffsetX=-0.000897892, viewOffsetY=0.000643996)
session.viewports['Viewport: 1'].view.setValues(width=0.128956, 
    height=0.0929196, viewOffsetX=-0.000948576, viewOffsetY=0.000634226)
session.viewports['Viewport: 2'].view.setValues(width=0.12884, 
    height=0.0928361, viewOffsetX=-0.000947724, viewOffsetY=0.000633655)
session.viewports['Viewport: 3'].view.setValues(width=0.12921, 
    height=0.0931029, viewOffsetX=-0.000950447, viewOffsetY=0.000635477)
session.viewports['Viewport: 4'].view.setValues(width=0.129316, 
    height=0.0931794, viewOffsetX=-0.000951229, viewOffsetY=0.000636)
session.viewports['Viewport: 1'].view.setValues(width=0.122408, 
    height=0.0882017, viewOffsetX=-0.000749347, viewOffsetY=0.000460183)
session.viewports['Viewport: 2'].view.setValues(width=0.122298, 
    height=0.0881224, viewOffsetX=-0.000748674, viewOffsetY=0.000459769)
session.viewports['Viewport: 3'].view.setValues(width=0.12265, 
    height=0.0883756, viewOffsetX=-0.000750823, viewOffsetY=0.000461091)
session.viewports['Viewport: 4'].view.setValues(width=0.12275, 
    height=0.0884482, viewOffsetX=-0.000751443, viewOffsetY=0.00046147)
session.viewports['Viewport: 1'].view.setValues(width=0.116136, 
    height=0.0836822, viewOffsetX=-0.000558476, viewOffsetY=0.000293441)
session.viewports['Viewport: 2'].view.setValues(width=0.116032, 
    height=0.083607, viewOffsetX=-0.000557974, viewOffsetY=0.000293177)
session.viewports['Viewport: 3'].view.setValues(width=0.116365, 
    height=0.0838472, viewOffsetX=-0.000559576, viewOffsetY=0.00029402)
session.viewports['Viewport: 4'].view.setValues(width=0.116461, 
    height=0.0839162, viewOffsetX=-0.000560038, viewOffsetY=0.000294262)
session.viewports['Viewport: 1'].view.setValues(width=0.110133, 
    height=0.0793571, viewOffsetX=-0.000375788, viewOffsetY=0.000133846)
session.viewports['Viewport: 2'].view.setValues(width=0.110035, 
    height=0.0792857, viewOffsetX=-0.00037545, viewOffsetY=0.000133725)
session.viewports['Viewport: 3'].view.setValues(width=0.11035, 
    height=0.0795134, viewOffsetX=-0.000376526, viewOffsetY=0.00013411)
session.viewports['Viewport: 4'].view.setValues(width=0.110441, 
    height=0.0795789, viewOffsetX=-0.000376839, viewOffsetY=0.000134221)
session.viewports['Viewport: 1'].view.setValues(width=0.104394, 
    height=0.0752215, viewOffsetX=-0.00020109, viewOffsetY=-1.87696e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.1043, height=0.0751539, 
    viewOffsetX=-0.00020091, viewOffsetY=-1.87536e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.1046, height=0.0753697, 
    viewOffsetX=-0.000201485, viewOffsetY=-1.88067e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.104686, 
    height=0.0754318, viewOffsetX=-0.000201653, viewOffsetY=-1.88216e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.0989112, 
    height=0.0712708, viewOffsetX=-1.17135e-05, viewOffsetY=-0.000164583)
session.viewports['Viewport: 2'].view.setValues(width=0.0988223, 
    height=0.0712067, viewOffsetX=-1.17025e-05, viewOffsetY=-0.000164436)
session.viewports['Viewport: 3'].view.setValues(width=0.0991059, 
    height=0.0714111, viewOffsetX=-1.17347e-05, viewOffsetY=-0.000164907)
session.viewports['Viewport: 4'].view.setValues(width=0.0991877, 
    height=0.07147, viewOffsetX=-1.1747e-05, viewOffsetY=-0.000165042)
session.viewports['Viewport: 1'].view.setValues(width=0.0936776, 
    height=0.0674997, viewOffsetX=0.00016907, viewOffsetY=-0.00030378)
session.viewports['Viewport: 2'].view.setValues(width=0.0935934, 
    height=0.0674391, viewOffsetX=0.000168919, viewOffsetY=-0.000303508)
session.viewports['Viewport: 3'].view.setValues(width=0.0938619, 
    height=0.0676325, viewOffsetX=0.000169405, viewOffsetY=-0.000304379)
session.viewports['Viewport: 4'].view.setValues(width=0.0939395, 
    height=0.0676884, viewOffsetX=0.000169542, viewOffsetY=-0.000304629)
session.viewports['Viewport: 1'].view.setValues(width=0.088686, 
    height=0.063903, viewOffsetX=0.000341512, viewOffsetY=-0.000436556)
session.viewports['Viewport: 2'].view.setValues(width=0.0886063, 
    height=0.0638456, viewOffsetX=0.000341206, viewOffsetY=-0.000436165)
session.viewports['Viewport: 3'].view.setValues(width=0.0888604, 
    height=0.0640287, viewOffsetX=0.000342186, viewOffsetY=-0.000437415)
session.viewports['Viewport: 4'].view.setValues(width=0.088934, 
    height=0.0640817, viewOffsetX=0.000342467, viewOffsetY=-0.000437776)
session.viewports['Viewport: 1'].view.setValues(width=0.0839288, 
    height=0.0604752, viewOffsetX=0.000505872, viewOffsetY=-0.000563109)
session.viewports['Viewport: 2'].view.setValues(width=0.0838534, 
    height=0.0604208, viewOffsetX=0.000505418, viewOffsetY=-0.000562604)
session.viewports['Viewport: 3'].view.setValues(width=0.0840938, 
    height=0.0605941, viewOffsetX=0.000506869, viewOffsetY=-0.000564217)
session.viewports['Viewport: 4'].view.setValues(width=0.0841635, 
    height=0.0606443, viewOffsetX=0.000507286, viewOffsetY=-0.000564683)
session.viewports['Viewport: 1'].view.setValues(width=0.0793983, 
    height=0.0572107, viewOffsetX=0.000644378, viewOffsetY=-0.000683643)
session.viewports['Viewport: 2'].view.setValues(width=0.079327, 
    height=0.0571593, viewOffsetX=0.000643799, viewOffsetY=-0.00068303)
session.viewports['Viewport: 3'].view.setValues(width=0.0795544, 
    height=0.0573232, viewOffsetX=0.000645647, viewOffsetY=-0.000684987)
session.viewports['Viewport: 4'].view.setValues(width=0.0796203, 
    height=0.0573707, viewOffsetX=0.000646179, viewOffsetY=-0.000685554)
session.viewports['Viewport: 1'].view.setValues(width=0.0750867, 
    height=0.054104, viewOffsetX=0.000605642, viewOffsetY=-0.000645255)
session.viewports['Viewport: 2'].view.setValues(width=0.0750193, 
    height=0.0540554, viewOffsetX=0.000605098, viewOffsetY=-0.000644677)
session.viewports['Viewport: 3'].view.setValues(width=0.0752342, 
    height=0.0542103, viewOffsetX=0.000606835, viewOffsetY=-0.000646524)
session.viewports['Viewport: 4'].view.setValues(width=0.0752966, 
    height=0.0542552, viewOffsetX=0.000607335, viewOffsetY=-0.00064706)
session.viewports['Viewport: 1'].view.setValues(width=0.0709861, 
    height=0.0511493, viewOffsetX=0.000568803, viewOffsetY=-0.000608747)
session.viewports['Viewport: 2'].view.setValues(width=0.0709224, 
    height=0.0511033, viewOffsetX=0.000568292, viewOffsetY=-0.000608201)
session.viewports['Viewport: 3'].view.setValues(width=0.0711255, 
    height=0.0512497, viewOffsetX=0.000569922, viewOffsetY=-0.000609943)
session.viewports['Viewport: 4'].view.setValues(width=0.0711846, 
    height=0.0512923, viewOffsetX=0.000570392, viewOffsetY=-0.000610449)
session.viewports['Viewport: 1'].view.setValues(width=0.0670887, 
    height=0.048341, viewOffsetX=0.000533788, viewOffsetY=-0.000574047)
session.viewports['Viewport: 2'].view.setValues(width=0.0670284, 
    height=0.0482975, viewOffsetX=0.000533308, viewOffsetY=-0.000573532)
session.viewports['Viewport: 3'].view.setValues(width=0.0672204, 
    height=0.0484359, viewOffsetX=0.000534838, viewOffsetY=-0.000575175)
session.viewports['Viewport: 4'].view.setValues(width=0.0672762, 
    height=0.0484761, viewOffsetX=0.000535279, viewOffsetY=-0.000575652)
session.viewports['Viewport: 1'].view.setValues(width=0.0633865, 
    height=0.0456734, viewOffsetX=0.000500527, viewOffsetY=-0.000541085)
session.viewports['Viewport: 2'].view.setValues(width=0.0633296, 
    height=0.0456323, viewOffsetX=0.000500077, viewOffsetY=-0.0005406)
session.viewports['Viewport: 3'].view.setValues(width=0.0635109, 
    height=0.045763, viewOffsetX=0.000501512, viewOffsetY=-0.000542148)
session.viewports['Viewport: 4'].view.setValues(width=0.0635637, 
    height=0.0458011, viewOffsetX=0.000501926, viewOffsetY=-0.000542598)
session.viewports['Viewport: 1'].view.setValues(width=0.0598719, 
    height=0.0431409, viewOffsetX=0.000428151, viewOffsetY=-0.000509793)
session.viewports['Viewport: 2'].view.setValues(width=0.0598181, 
    height=0.0431021, viewOffsetX=0.000427766, viewOffsetY=-0.000509336)
session.viewports['Viewport: 3'].view.setValues(width=0.0599894, 
    height=0.0432255, viewOffsetX=0.000428994, viewOffsetY=-0.000510795)
session.viewports['Viewport: 4'].view.setValues(width=0.0600393, 
    height=0.0432615, viewOffsetX=0.000429347, viewOffsetY=-0.000511219)
session.viewports['Viewport: 1'].view.setValues(width=0.0565371, 
    height=0.040738, viewOffsetX=0.000359476, viewOffsetY=-0.000480102)
session.viewports['Viewport: 2'].view.setValues(width=0.0564863, 
    height=0.0407014, viewOffsetX=0.000359153, viewOffsetY=-0.000479672)
session.viewports['Viewport: 3'].view.setValues(width=0.056648, 
    height=0.0408179, viewOffsetX=0.000360184, viewOffsetY=-0.000481045)
session.viewports['Viewport: 4'].view.setValues(width=0.0566951, 
    height=0.0408519, viewOffsetX=0.000360481, viewOffsetY=-0.000481445)
session.viewports['Viewport: 1'].view.setValues(width=0.0533745, 
    height=0.0384592, viewOffsetX=0.000294347, viewOffsetY=-0.000451945)
session.viewports['Viewport: 2'].view.setValues(width=0.0533266, 
    height=0.0384246, viewOffsetX=0.000294082, viewOffsetY=-0.00045154)
session.viewports['Viewport: 3'].view.setValues(width=0.0534792, 
    height=0.0385346, viewOffsetX=0.000294926, viewOffsetY=-0.000452833)
session.viewports['Viewport: 4'].view.setValues(width=0.0535237, 
    height=0.0385667, viewOffsetX=0.000295169, viewOffsetY=-0.000453209)
session.viewports['Viewport: 1'].view.setValues(width=0.0503768, 
    height=0.0362992, viewOffsetX=0.00023261, viewOffsetY=-0.000425255)
session.viewports['Viewport: 2'].view.setValues(width=0.0503315, 
    height=0.0362666, viewOffsetX=0.000232401, viewOffsetY=-0.000424874)
session.viewports['Viewport: 3'].view.setValues(width=0.0504756, 
    height=0.0363703, viewOffsetX=0.000233068, viewOffsetY=-0.00042609)
session.viewports['Viewport: 4'].view.setValues(width=0.0505176, 
    height=0.0364006, viewOffsetX=0.00023326, viewOffsetY=-0.000426444)
session.viewports['Viewport: 1'].view.setValues(width=0.0475366, 
    height=0.0342527, viewOffsetX=0.000141723, viewOffsetY=-0.000335347)
session.viewports['Viewport: 2'].view.setValues(width=0.0474939, 
    height=0.0342219, viewOffsetX=0.000141595, viewOffsetY=-0.000335047)
session.viewports['Viewport: 3'].view.setValues(width=0.0476298, 
    height=0.0343198, viewOffsetX=0.000142003, viewOffsetY=-0.000336006)
session.viewports['Viewport: 4'].view.setValues(width=0.0476695, 
    height=0.0343484, viewOffsetX=0.000142118, viewOffsetY=-0.000336285)
session.viewports['Viewport: 1'].view.setValues(width=0.0448469, 
    height=0.0323146, viewOffsetX=5.56478e-05, viewOffsetY=-0.0002502)
session.viewports['Viewport: 2'].view.setValues(width=0.0448066, 
    height=0.0322855, viewOffsetX=5.55977e-05, viewOffsetY=-0.000249977)
session.viewports['Viewport: 3'].view.setValues(width=0.0449348, 
    height=0.0323779, viewOffsetX=5.57591e-05, viewOffsetY=-0.000250692)
session.viewports['Viewport: 4'].view.setValues(width=0.0449723, 
    height=0.0324049, viewOffsetX=5.5803e-05, viewOffsetY=-0.0002509)
session.viewports['Viewport: 1'].view.setValues(width=0.0423007, 
    height=0.0304799, viewOffsetX=-2.58352e-05, viewOffsetY=-0.000169595)
session.viewports['Viewport: 2'].view.setValues(width=0.0422627, 
    height=0.0304525, viewOffsetX=-2.58122e-05, viewOffsetY=-0.000169444)
session.viewports['Viewport: 3'].view.setValues(width=0.0423836, 
    height=0.0305396, viewOffsetX=-2.58837e-05, viewOffsetY=-0.000169929)
session.viewports['Viewport: 4'].view.setValues(width=0.0424189, 
    height=0.0305651, viewOffsetX=-2.59079e-05, viewOffsetY=-0.00017007)
session.viewports['Viewport: 1'].view.setValues(width=0.0398913, 
    height=0.0287438, viewOffsetX=-0.000102941, viewOffsetY=-9.33212e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0398555, 
    height=0.028718, viewOffsetX=-0.000102849, viewOffsetY=-9.32389e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.0399695, 
    height=0.0288001, viewOffsetX=-0.00010314, viewOffsetY=-9.35056e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0400029, 
    height=0.0288242, viewOffsetX=-0.000103229, viewOffsetY=-9.35823e-05)
session.viewports['Viewport: 1'].setValues(origin=(0.0, 81.7569351196289), 
    height=68.8009262084961)
session.viewports['Viewport: 1'].setValues(origin=(0.0, 115.934020996094), 
    width=130.791656494141, height=34.6238403320312)
session.viewports['Viewport: 2'].setValues(origin=(130.791656494141, 
    115.934020996094), width=130.791656494141, height=34.6238403320312)
session.viewports['Viewport: 3'].setValues(origin=(0.0, 81.310188293457), 
    width=130.791656494141, height=34.6238403320312)
session.viewports['Viewport: 4'].setValues(origin=(130.791656494141, 
    81.310188293457), width=130.791656494141, height=34.6238403320312)
session.viewports['Viewport: 5'].setValues(origin=(0.0, 46.6863403320312), 
    width=130.791656494141, height=34.6238403320312)
session.viewports['Viewport: 6'].setValues(origin=(130.791656494141, 
    46.6863403320312), width=130.791656494141, height=34.6238403320312)
session.viewports['Viewport: 7'].setValues(origin=(0.0, 12.0625), 
    width=130.791656494141, height=34.6238403320312)
session.viewports['Viewport: 8'].setValues(origin=(130.791656494141, 12.0625), 
    width=130.791656494141, height=34.6238403320312)
session.viewports['Viewport: 1'].setValues(origin=(0.0, 81.3101806640625), 
    width=65.3958282470703, height=69.2476806640625)
session.viewports['Viewport: 2'].setValues(origin=(65.3958282470703, 
    81.3101806640625), width=65.3958282470703, height=69.2476806640625)
session.viewports['Viewport: 3'].setValues(origin=(130.791656494141, 
    81.3101806640625), width=65.3958282470703, height=69.2476806640625)
session.viewports['Viewport: 4'].setValues(origin=(196.1875, 81.3101806640625), 
    width=65.3958282470703, height=69.2476806640625)
session.viewports['Viewport: 5'].setValues(origin=(0.0, 12.0625), 
    width=65.3958282470703, height=69.2476806640625)
session.viewports['Viewport: 6'].setValues(origin=(65.3958282470703, 12.0625), 
    width=65.3958282470703, height=69.2476806640625)
session.viewports['Viewport: 7'].setValues(origin=(130.791656494141, 12.0625), 
    width=65.3958282470703, height=69.2476806640625)
session.viewports['Viewport: 8'].setValues(origin=(196.1875, 12.0625), 
    width=65.3958282470703, height=69.2476806640625)
session.viewports['Viewport: 1'].view.setValues(width=0.0425672, 
    height=0.0420039, viewOffsetX=-0.000153245, viewOffsetY=-0.000148)
session.viewports['Viewport: 2'].view.setValues(width=0.0425289, 
    height=0.0419662, viewOffsetX=-0.000153108, viewOffsetY=-0.000147868)
session.viewports['Viewport: 3'].view.setValues(width=0.0426505, 
    height=0.0420862, viewOffsetX=-0.000153543, viewOffsetY=-0.000148291)
session.viewports['Viewport: 4'].view.setValues(width=0.0426862, 
    height=0.0421213, viewOffsetX=-0.000153674, viewOffsetY=-0.000148414)
session.viewports['Viewport: 1'].view.setValues(width=0.0451311, 
    height=0.0445339, viewOffsetX=-0.000205728, viewOffsetY=-0.000205169)
session.viewports['Viewport: 2'].view.setValues(width=0.0450906, 
    height=0.0444939, viewOffsetX=-0.000205543, viewOffsetY=-0.000204986)
session.viewports['Viewport: 3'].view.setValues(width=0.0452196, 
    height=0.0446212, viewOffsetX=-0.000206129, viewOffsetY=-0.000205573)
session.viewports['Viewport: 4'].view.setValues(width=0.0452573, 
    height=0.0446585, viewOffsetX=-0.000206303, viewOffsetY=-0.000205743)
session.viewports['Viewport: 1'].view.setValues(width=0.0422854, 
    height=0.0417259, viewOffsetX=-0.000181046, viewOffsetY=-0.000175761)
session.viewports['Viewport: 2'].view.setValues(width=0.0422474, 
    height=0.0416884, viewOffsetX=-0.000180884, viewOffsetY=-0.000175605)
session.viewports['Viewport: 3'].view.setValues(width=0.0423683, 
    height=0.0418076, viewOffsetX=-0.000181399, viewOffsetY=-0.000176107)
session.viewports['Viewport: 4'].view.setValues(width=0.0424036, 
    height=0.0418425, viewOffsetX=-0.000181553, viewOffsetY=-0.000176253)
session.viewports['Viewport: 1'].view.setValues(width=0.0398921, 
    height=0.0393642, viewOffsetX=-0.000159047, viewOffsetY=-0.000149283)
session.viewports['Viewport: 2'].view.setValues(width=0.0398563, 
    height=0.0393289, viewOffsetX=-0.000158905, viewOffsetY=-0.00014915)
session.viewports['Viewport: 3'].view.setValues(width=0.0399703, 
    height=0.0394414, viewOffsetX=-0.000159357, viewOffsetY=-0.000149577)
session.viewports['Viewport: 4'].view.setValues(width=0.0400036, 
    height=0.0394743, viewOffsetX=-0.000159493, viewOffsetY=-0.0001497)
session.viewports['Viewport: 1'].view.setValues(width=0.0376123, 
    height=0.0371146, viewOffsetX=-0.00013817, viewOffsetY=-0.00012417)
session.viewports['Viewport: 2'].view.setValues(width=0.0375785, 
    height=0.0370813, viewOffsetX=-0.000138047, viewOffsetY=-0.00012406)
session.viewports['Viewport: 3'].view.setValues(width=0.037686, 
    height=0.0371873, viewOffsetX=-0.000138439, viewOffsetY=-0.000124415)
session.viewports['Viewport: 4'].view.setValues(width=0.0377175, 
    height=0.0372184, viewOffsetX=-0.000138557, viewOffsetY=-0.000124518)
session.viewports['Viewport: 1'].view.setValues(width=0.0354574, 
    height=0.0349882, viewOffsetX=-0.000118433, viewOffsetY=-0.000124528)
session.viewports['Viewport: 2'].view.setValues(width=0.0354255, 
    height=0.0349568, viewOffsetX=-0.000118327)
session.viewports['Viewport: 3'].view.setValues(width=0.0355268, 
    height=0.0350567, viewOffsetX=-0.000118663, viewOffsetY=-0.000124773)
session.viewports['Viewport: 4'].view.setValues(width=0.0355565, 
    height=0.035086, viewOffsetX=-0.000118764, viewOffsetY=-0.000124876)
session.viewports['Viewport: 1'].view.setValues(width=0.0334204, 
    height=0.0329782, viewOffsetX=-9.97756e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0333904, 
    height=0.0329486, viewOffsetX=-9.96862e-05, viewOffsetY=-0.000124755)
session.viewports['Viewport: 3'].view.setValues(width=0.0334858, 
    height=0.0330427, viewOffsetX=-9.99688e-05, viewOffsetY=-0.000125112)
session.viewports['Viewport: 4'].view.setValues(width=0.0335138, 
    height=0.0330704, viewOffsetX=-0.000100055)
session.viewports['Viewport: 1'].view.setValues(width=0.0314956, 
    height=0.0310788, viewOffsetX=-8.21452e-05, viewOffsetY=-0.000125185)
session.viewports['Viewport: 2'].view.setValues(width=0.0314673, 
    height=0.0310509, viewOffsetX=-8.20718e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.0315572, 
    height=0.0311396, viewOffsetX=-8.23038e-05, viewOffsetY=-0.000125432)
session.viewports['Viewport: 4'].view.setValues(width=0.0315836, 
    height=0.0311657, viewOffsetX=-8.23752e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.0296772, 
    height=0.0292845, viewOffsetX=-6.54902e-05, viewOffsetY=-0.000125487)
session.viewports['Viewport: 2'].view.setValues(width=0.0296506, 
    height=0.0292582, viewOffsetX=-6.54317e-05, viewOffsetY=-0.000125376)
session.viewports['Viewport: 3'].view.setValues(width=0.0297353, 
    height=0.0293419, viewOffsetX=-6.56162e-05, viewOffsetY=-0.000125734)
session.viewports['Viewport: 4'].view.setValues(width=0.0297602, 
    height=0.0293664, viewOffsetX=-6.56736e-05, viewOffsetY=-0.000125838)
session.viewports['Viewport: 1'].view.setValues(width=0.0316433, 
    height=0.0312246, viewOffsetX=-0.000136583, viewOffsetY=2.24882e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0316149, 
    height=0.0311965, viewOffsetX=-0.000136461, viewOffsetY=2.24668e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.0317052, 
    height=0.0312857, viewOffsetX=-0.000136848, viewOffsetY=2.25308e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0317318, 
    height=0.0313119, viewOffsetX=-0.000136965, viewOffsetY=2.25506e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.0296718, 
    height=0.0292791, viewOffsetX=-0.000105918, viewOffsetY=2.24331e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0296451, 
    height=0.0292528, viewOffsetX=-0.000105823, viewOffsetY=2.24117e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.0297298, 
    height=0.0293364, viewOffsetX=-0.000106123, viewOffsetY=2.24756e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0297547, 
    height=0.029361, viewOffsetX=-0.000106215, viewOffsetY=2.24953e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.0279602, 
    height=0.0275902, viewOffsetX=-7.75987e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0279351, 
    height=0.0275654, viewOffsetX=-7.75292e-05, viewOffsetY=2.2467e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.0280149, 
    height=0.0276442, viewOffsetX=-7.77484e-05, viewOffsetY=2.25311e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0280384, 
    height=0.0276673, viewOffsetX=-7.78159e-05, viewOffsetY=2.25508e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.0263387, 
    height=0.0259902, viewOffsetX=-5.08413e-05, viewOffsetY=2.25364e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.026315, 
    height=0.0259668, viewOffsetX=-5.07958e-05, viewOffsetY=2.2515e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.0263902, 
    height=0.026041, viewOffsetX=-5.09387e-05, viewOffsetY=2.25792e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0264123, 
    height=0.0260628, viewOffsetX=-5.09837e-05, viewOffsetY=2.2599e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.0248083, 
    height=0.02448, viewOffsetX=-2.55851e-05, viewOffsetY=2.2582e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.024786, 
    height=0.0244581, viewOffsetX=-2.55624e-05, viewOffsetY=2.25605e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.0248569, 
    height=0.024528, viewOffsetX=-2.56331e-05, viewOffsetY=2.26248e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0248777, 
    height=0.0245485, viewOffsetX=-2.5657e-05, viewOffsetY=2.26447e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.0233641, 
    height=0.023055, viewOffsetX=-2.82874e-05, viewOffsetY=4.90925e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0233432, 
    height=0.0230343, viewOffsetX=-2.82623e-05, viewOffsetY=4.90472e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.0234099, 
    height=0.0231001, viewOffsetX=-2.83406e-05, viewOffsetY=4.91871e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0234295, 
    height=0.0231194, viewOffsetX=-2.83667e-05, viewOffsetY=4.92293e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.0220016, 
    height=0.0217105, viewOffsetX=-3.08368e-05, viewOffsetY=7.41044e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0219819, 
    height=0.021691, viewOffsetX=-3.08095e-05, viewOffsetY=7.40367e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.0220447, 
    height=0.021753, viewOffsetX=-3.08951e-05, viewOffsetY=7.4248e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0220631, 
    height=0.0217712, viewOffsetX=-3.09233e-05, viewOffsetY=7.43111e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.0207164, 
    height=0.0204423, viewOffsetX=-3.32417e-05, viewOffsetY=9.76975e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0206978, 
    height=0.0204239, viewOffsetX=-3.32122e-05, viewOffsetY=9.76085e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.0207569, 
    height=0.0204822, viewOffsetX=-3.33046e-05, viewOffsetY=9.78871e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0207743, 
    height=0.0204994, viewOffsetX=-3.33349e-05, viewOffsetY=9.79701e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.0195043, 
    height=0.0192462, viewOffsetX=-3.55097e-05, viewOffsetY=0.000119948)
session.viewports['Viewport: 2'].view.setValues(width=0.0194868, 
    height=0.0192289, viewOffsetX=-3.54782e-05, viewOffsetY=0.000119839)
session.viewports['Viewport: 3'].view.setValues(width=0.0195425, 
    height=0.0192839, viewOffsetX=-3.5577e-05, viewOffsetY=0.000120181)
session.viewports['Viewport: 4'].view.setValues(width=0.0195588, height=0.0193, 
    viewOffsetX=-3.56092e-05, viewOffsetY=0.000120282)
session.viewports['Viewport: 1'].view.setValues(width=0.0183615, 
    height=0.0181185, viewOffsetX=-1.67939e-05, viewOffsetY=0.000120127)
session.viewports['Viewport: 2'].view.setValues(width=0.018345, 
    height=0.0181022, viewOffsetX=-1.67793e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.0183974, 
    height=0.0181539, viewOffsetX=-1.68246e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0184128, 
    height=0.0181691, viewOffsetX=-1.68412e-05, viewOffsetY=0.000120462)
session.viewports['Viewport: 1'].view.setValues(width=0.0172841, 
    height=0.0170554, viewOffsetX=8.50006e-07)
session.viewports['Viewport: 2'].view.setValues(width=0.0172685, 
    height=0.01704, viewOffsetX=8.48813e-07)
session.viewports['Viewport: 3'].view.setValues(width=0.0173179, 
    height=0.0170887, viewOffsetX=8.53819e-07)
session.viewports['Viewport: 4'].view.setValues(width=0.0173324, 
    height=0.017103, viewOffsetX=8.52015e-07, viewOffsetY=0.000120632)
session.viewports['Viewport: 1'].view.setValues(width=0.0162685, 
    height=0.0160533, viewOffsetX=1.74806e-05, viewOffsetY=0.000120455)
session.viewports['Viewport: 2'].view.setValues(width=0.0162539, 
    height=0.0160389, viewOffsetX=1.74645e-05, viewOffsetY=0.000120346)
session.viewports['Viewport: 3'].view.setValues(width=0.0163004, 
    height=0.0160847, viewOffsetX=1.7517e-05, viewOffsetY=0.000120689)
session.viewports['Viewport: 4'].view.setValues(width=0.016314, 
    height=0.0160982, viewOffsetX=1.75291e-05)
session.viewports['Viewport: 1'].odbDisplay.basicOptions.setValues(
    mirrorAboutXyPlane=False)
session.viewports['Viewport: 3'].makeCurrent()
session.viewports['Viewport: 3'].odbDisplay.basicOptions.setValues(
    mirrorAboutXyPlane=False)
session.viewports['Viewport: 2'].makeCurrent()
session.viewports['Viewport: 2'].odbDisplay.basicOptions.setValues(
    mirrorAboutXyPlane=False)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 4'].makeCurrent()
session.viewports['Viewport: 4'].odbDisplay.basicOptions.setValues(
    mirrorAboutXyPlane=False)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM1', outputPosition=INTEGRATION_POINT, )
del session.viewports['Viewport: 5']
del session.viewports['Viewport: 6']
del session.viewports['Viewport: 7']
del session.viewports['Viewport: 8']
session.Viewport(name='Viewport: 10', origin=(21.5, 11.8391199111938), 
    width=227.09375, height=117.274307250977)
session.viewports['Viewport: 10'].makeCurrent()
session.Viewport(name='Viewport: 11', origin=(26.875, 11.8391199111938), 
    width=227.09375, height=111.913192749023)
session.viewports['Viewport: 11'].makeCurrent()
session.Viewport(name='Viewport: 12', origin=(32.25, 11.8391199111938), 
    width=227.09375, height=106.55207824707)
session.viewports['Viewport: 12'].makeCurrent()
session.viewports['Viewport: 1'].setValues(width=65.3958282470703, 
    height=69.2476806640625)
session.viewports['Viewport: 2'].setValues(origin=(65.3958282470703, 
    81.3101806640625), width=65.3958282470703, height=69.2476806640625)
session.viewports['Viewport: 3'].setValues(origin=(130.791656494141, 
    81.3101806640625), width=65.3958282470703, height=69.2476806640625)
session.viewports['Viewport: 4'].setValues(width=65.3958282470703, 
    height=69.2476806640625)
session.viewports['Viewport: 10'].setValues(origin=(0.0, 12.0625), 
    width=65.3958282470703, height=69.2476806640625)
session.viewports['Viewport: 11'].setValues(origin=(65.3958282470703, 12.0625), 
    width=65.3958282470703, height=69.2476806640625)
session.viewports['Viewport: 12'].setValues(origin=(130.791656494141, 12.0625), 
    width=65.3958282470703, height=69.2476806640625)
session.viewports['Viewport: 2'].setValues(origin=(64.7239532470703, 
    81.3101806640625), width=65.6197891235352)
session.viewports['Viewport: 1'].setValues(origin=(0.0, 115.934020996094), 
    width=143.33332824707, height=34.6238403320312)
session.viewports['Viewport: 2'].setValues(origin=(143.33332824707, 
    115.934020996094), width=143.33332824707, height=34.6238403320312)
session.viewports['Viewport: 3'].setValues(origin=(0.0, 81.310188293457), 
    width=143.33332824707, height=34.6238403320312)
session.viewports['Viewport: 4'].setValues(origin=(143.33332824707, 
    81.310188293457), width=143.33332824707, height=34.6238403320312)
session.viewports['Viewport: 10'].setValues(origin=(0.0, 46.6863403320312), 
    width=143.33332824707, height=34.6238403320312)
session.viewports['Viewport: 11'].setValues(origin=(143.33332824707, 
    46.6863403320312), width=143.33332824707, height=34.6238403320312)
session.viewports['Viewport: 12'].setValues(origin=(0.0, 12.0625), 
    width=143.33332824707, height=34.6238403320312)
session.viewports['Viewport: 1'].setValues(origin=(0.0, 81.3101806640625), 
    width=71.6666641235352, height=69.2476806640625)
session.viewports['Viewport: 2'].setValues(origin=(71.6666641235352, 
    81.3101806640625), width=71.6666641235352, height=69.2476806640625)
session.viewports['Viewport: 3'].setValues(origin=(143.33332824707, 
    81.3101806640625), width=71.6666641235352, height=69.2476806640625)
session.viewports['Viewport: 4'].setValues(origin=(215.0, 81.3101806640625), 
    width=71.6666641235352, height=69.2476806640625)
session.viewports['Viewport: 10'].setValues(origin=(0.0, 12.0625), 
    width=71.6666641235352, height=69.2476806640625)
session.viewports['Viewport: 11'].setValues(origin=(71.6666641235352, 12.0625), 
    width=71.6666641235352, height=69.2476806640625)
session.viewports['Viewport: 12'].setValues(origin=(143.33332824707, 12.0625), 
    width=71.6666641235352, height=69.2476806640625)
session.viewports['Viewport: 10'].setValues(origin=(80.1770782470703, 
    11.6157379150391))
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].setValues(origin=(0.0, 11.3923645019531), 
    height=138.942123413086)
session.viewports['Viewport: 11'].setValues(origin=(71.4427032470703, 12.0625), 
    width=71.890625, height=69.2476806640625)
session.viewports['Viewport: 11'].makeCurrent()
session.viewports['Viewport: 1'].setValues(width=49.046875)
session.viewports['Viewport: 1'].setValues(width=38.0729141235352)
session.viewports['Viewport: 2'].setValues(origin=(38.296875, 
    81.3101806640625), width=105.03645324707)
session.viewports['Viewport: 2'].setValues(width=46.8072891235352)
session.viewports['Viewport: 2'].setValues(origin=(38.296875, 
    11.6157379150391), height=138.71875)
session.viewports['Viewport: 12'].setValues(origin=(187.453125, 
    10.0520782470703))
session.viewports['Viewport: 11'].setValues(origin=(204.25, 11.6157379150391))
session.viewports['Viewport: 2'].setValues(width=33.8177070617676)
session.viewports['Viewport: 2'].setValues(width=28.4427070617676)
#* RangeError: width must be a Float in the range: 30 <= width <= 3350
session.viewports['Viewport: 1'].setValues(width=21.9479160308838)
#* RangeError: width must be a Float in the range: 30 <= width <= 3350
session.viewports['Viewport: 2'].setValues(width=17.9166660308838)
#* RangeError: width must be a Float in the range: 30 <= width <= 3350
session.viewports['Viewport: 1'].view.setValues(width=0.015096, 
    height=0.0556167, viewOffsetX=-6.10342e-05, viewOffsetY=0.000121912)
session.viewports['Viewport: 2'].view.setValues(width=0.0150102, 
    height=0.0626462, viewOffsetX=-6.62496e-06)
session.viewports['Viewport: 3'].view.setValues(width=0.0169734, 
    height=0.015182, viewOffsetX=-0.000289959, viewOffsetY=-0.000422604)
session.viewports['Viewport: 4'].view.setValues(width=0.0169892, 
    height=0.0151961, viewOffsetX=-0.000290231, viewOffsetY=-0.000422995)
session.viewports['Viewport: 10'].view.setValues(width=0.0168797, 
    height=0.0151524, viewOffsetX=-0.000288297, viewOffsetY=-0.000418101)
session.viewports['Viewport: 11'].view.setValues(width=0.0169351, 
    height=0.0151533, viewOffsetX=-0.000290052, viewOffsetY=-0.000418126)
session.viewports['Viewport: 12'].view.setValues(width=0.0168797, 
    height=0.0151524, viewOffsetX=-0.000288297, viewOffsetY=-0.000418101)
session.viewports['Viewport: 1'].view.setValues(width=0.0142266, 
    height=0.0524137, viewOffsetX=-0.000134972, viewOffsetY=0.000125055)
session.viewports['Viewport: 2'].view.setValues(width=0.0141533, 
    height=0.0590701, viewOffsetX=-2.90704e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.0159431, 
    height=0.0142605, viewOffsetX=-0.000578624, viewOffsetY=-0.000933073)
session.viewports['Viewport: 4'].view.setValues(width=0.0159579, 
    height=0.0142737, viewOffsetX=-0.000579163, viewOffsetY=-0.000933935)
session.viewports['Viewport: 10'].view.setValues(width=0.0158561, 
    height=0.0142337, viewOffsetX=-0.000575406, viewOffsetY=-0.000924186)
session.viewports['Viewport: 11'].view.setValues(width=0.0159073, 
    height=0.0142336, viewOffsetX=-0.000578774, viewOffsetY=-0.000924184)
session.viewports['Viewport: 12'].view.setValues(width=0.0158561, 
    height=0.0142337, viewOffsetX=-0.000575406, viewOffsetY=-0.000924186)
session.viewports['Viewport: 1'].view.setValues(width=0.0151633, 
    height=0.0558649, viewOffsetX=-6.13065e-05, viewOffsetY=0.000122456)
session.viewports['Viewport: 2'].view.setValues(width=0.0150888, 
    height=0.0629744, viewOffsetX=-6.65968e-06)
session.viewports['Viewport: 3'].view.setValues(width=0.0169801, 
    height=0.015188, viewOffsetX=-0.000290074, viewOffsetY=-0.000422772)
session.viewports['Viewport: 4'].view.setValues(width=0.0169959, 
    height=0.0152021, viewOffsetX=-0.000290345, viewOffsetY=-0.000423161)
session.viewports['Viewport: 10'].view.setValues(width=0.0168874, 
    height=0.0151594, viewOffsetX=-0.000288429, viewOffsetY=-0.000418293)
session.viewports['Viewport: 11'].view.setValues(width=0.0169419, 
    height=0.0151594, viewOffsetX=-0.000290168, viewOffsetY=-0.000418292)
session.viewports['Viewport: 12'].view.setValues(width=0.0168874, 
    height=0.0151594, viewOffsetX=-0.000288429, viewOffsetY=-0.000418293)
session.viewports['Viewport: 1'].view.setValues(width=0.0160983, 
    height=0.0593096, viewOffsetX=1.72979e-05, viewOffsetY=0.000119195)
session.viewports['Viewport: 2'].view.setValues(width=0.0160147, 
    height=0.0668385, viewOffsetX=1.72074e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.0180433, 
    height=0.0161389, viewOffsetX=1.75759e-05, viewOffsetY=0.000121095)
session.viewports['Viewport: 4'].view.setValues(width=0.01806, 
    height=0.0161539, viewOffsetX=1.75894e-05, viewOffsetY=0.000121209)
session.viewports['Viewport: 10'].view.setValues(width=0.0179448, 
    height=0.0161086, viewOffsetX=1.75405e-05, viewOffsetY=0.000120869)
session.viewports['Viewport: 11'].view.setValues(width=0.0180027, 
    height=0.0161086, viewOffsetX=1.75405e-05, viewOffsetY=0.00012087)
session.viewports['Viewport: 12'].view.setValues(width=0.0179448, 
    height=0.0161086, viewOffsetX=1.75405e-05, viewOffsetY=0.000120869)
session.viewports['Viewport: 1'].view.setValues(width=0.017091, 
    height=0.0629667, viewOffsetX=0.000100581, viewOffsetY=9.65785e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0169977, 
    height=0.0709412, viewOffsetX=4.24835e-05, viewOffsetY=9.66584e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.0191724, 
    height=0.0171489, viewOffsetX=0.000344104, viewOffsetY=0.000687235)
session.viewports['Viewport: 4'].view.setValues(width=0.0191901, 
    height=0.0171647, viewOffsetX=0.00034442, viewOffsetY=0.000687873)
session.viewports['Viewport: 10'].view.setValues(width=0.0190678, 
    height=0.0171167, viewOffsetX=0.000342288, viewOffsetY=0.00068208)
session.viewports['Viewport: 11'].view.setValues(width=0.0191293, 
    height=0.0171167, viewOffsetX=0.000344133, viewOffsetY=0.000682081)
session.viewports['Viewport: 12'].view.setValues(width=0.0190678, 
    height=0.0171167, viewOffsetX=0.000342288, viewOffsetY=0.00068208)
session.viewports['Viewport: 1'].view.setValues(width=0.0181425, 
    height=0.0668407, viewOffsetX=0.000188808, viewOffsetY=7.26185e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0180383, 
    height=0.0752844, viewOffsetX=6.9245e-05, viewOffsetY=7.34536e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.0203706, 
    height=0.0182206, viewOffsetX=0.000690631, viewOffsetY=0.00128805)
session.viewports['Viewport: 4'].view.setValues(width=0.0203895, 
    height=0.0182375, viewOffsetX=0.000691267, viewOffsetY=0.00128924)
session.viewports['Viewport: 10'].view.setValues(width=0.0202596, 
    height=0.0181865, viewOffsetX=0.000686927, viewOffsetY=0.00127767)
session.viewports['Viewport: 11'].view.setValues(width=0.0203249, 
    height=0.0181865, viewOffsetX=0.000690731, viewOffsetY=0.00127767)
session.viewports['Viewport: 12'].view.setValues(width=0.0202596, 
    height=0.0181865, viewOffsetX=0.000686927, viewOffsetY=0.00127767)
session.viewports['Viewport: 1'].view.setValues(width=0.0170147, 
    height=0.0626856, viewOffsetX=0.000100132, viewOffsetY=9.61471e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0169119, 
    height=0.0705834, viewOffsetX=4.22692e-05, viewOffsetY=9.61709e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.0191229, 
    height=0.0171046, viewOffsetX=0.000343216, viewOffsetY=0.000685462)
session.viewports['Viewport: 4'].view.setValues(width=0.0191406, 
    height=0.0171204, viewOffsetX=0.000343531, viewOffsetY=0.000686098)
session.viewports['Viewport: 10'].view.setValues(width=0.0190188, 
    height=0.0170727, viewOffsetX=0.000341407, viewOffsetY=0.000680326)
session.viewports['Viewport: 11'].view.setValues(width=0.0190801, 
    height=0.0170727, viewOffsetX=0.000343249, viewOffsetY=0.000680328)
session.viewports['Viewport: 12'].view.setValues(width=0.0190188, 
    height=0.0170727, viewOffsetX=0.000341407, viewOffsetY=0.000680326)
session.viewports['Viewport: 1'].view.setValues(width=0.0160334, 
    height=0.0590703, viewOffsetX=1.72278e-05, viewOffsetY=0.000118714)
session.viewports['Viewport: 2'].view.setValues(width=0.0159421, 
    height=0.0665356, viewOffsetX=1.71294e-05, viewOffsetY=0.000118037)
session.viewports['Viewport: 3'].view.setValues(width=0.0180005, 
    height=0.0161006, viewOffsetX=1.7534e-05, viewOffsetY=0.000120807)
session.viewports['Viewport: 4'].view.setValues(width=0.0180171, 
    height=0.0161155, viewOffsetX=1.7548e-05, viewOffsetY=0.000120922)
session.viewports['Viewport: 10'].view.setValues(width=0.0179024, 
    height=0.0160705, viewOffsetX=1.74986e-05, viewOffsetY=0.000120582)
session.viewports['Viewport: 11'].view.setValues(width=0.0179601, 
    height=0.0160705, viewOffsetX=1.75e-05, viewOffsetY=0.000120585)
session.viewports['Viewport: 12'].view.setValues(width=0.0179024, 
    height=0.0160705, viewOffsetX=1.74986e-05, viewOffsetY=0.000120582)
session.viewports['Viewport: 1'].view.setValues(width=0.0151038, 
    height=0.0556453, viewOffsetX=-6.10658e-05, viewOffsetY=0.000140004)
session.viewports['Viewport: 2'].view.setValues(width=0.0150218, 
    height=0.0626949, viewOffsetX=-6.63014e-06, viewOffsetY=0.000138671)
session.viewports['Viewport: 3'].view.setValues(width=0.0169415, 
    height=0.0151535, viewOffsetX=-0.000289415, viewOffsetY=-0.000411373)
session.viewports['Viewport: 4'].view.setValues(width=0.0169572, 
    height=0.0151675, viewOffsetX=-0.000289685, viewOffsetY=-0.000411751)
session.viewports['Viewport: 10'].view.setValues(width=0.0168491, 
    height=0.015125, viewOffsetX=-0.000287775, viewOffsetY=-0.000406966)
session.viewports['Viewport: 11'].view.setValues(width=0.0169035, 
    height=0.015125, viewOffsetX=-0.000289508, viewOffsetY=-0.000406962)
session.viewports['Viewport: 12'].view.setValues(width=0.0168491, 
    height=0.015125, viewOffsetX=-0.000287775, viewOffsetY=-0.000406966)
session.viewports['Viewport: 1'].view.setValues(width=0.0142263, 
    height=0.0524128, viewOffsetX=-0.00013497, viewOffsetY=0.0001601)
session.viewports['Viewport: 2'].view.setValues(width=0.0141529, 
    height=0.0590684, viewOffsetX=-2.90696e-05, viewOffsetY=0.00015816)
session.viewports['Viewport: 3'].view.setValues(width=0.0159437, 
    height=0.014261, viewOffsetX=-0.000578646, viewOffsetY=-0.000912835)
session.viewports['Viewport: 4'].view.setValues(width=0.0159585, 
    height=0.0142742, viewOffsetX=-0.000579183, viewOffsetY=-0.000913677)
session.viewports['Viewport: 10'].view.setValues(width=0.0158567, 
    height=0.0142341, viewOffsetX=-0.000575427, viewOffsetY=-0.000904058)
session.viewports['Viewport: 11'].view.setValues(width=0.0159078, 
    height=0.0142341, viewOffsetX=-0.000578793, viewOffsetY=-0.000904053)
session.viewports['Viewport: 12'].view.setValues(width=0.0158567, 
    height=0.0142341, viewOffsetX=-0.000575427, viewOffsetY=-0.000904058)
session.viewports['Viewport: 1'].view.setValues(width=0.0133983, 
    height=0.0493622, viewOffsetX=-0.000204714, viewOffsetY=0.000179066)
session.viewports['Viewport: 2'].view.setValues(width=0.0133325, 
    height=0.0556442, viewOffsetX=-5.02567e-05, viewOffsetY=0.000176561)
session.viewports['Viewport: 3'].view.setValues(width=0.0150037, 
    height=0.0134201, viewOffsetX=-0.000851143, viewOffsetY=-0.00138528)
session.viewports['Viewport: 4'].view.setValues(width=0.0150175, 
    height=0.0134325, viewOffsetX=-0.000851932, viewOffsetY=-0.00138656)
session.viewports['Viewport: 10'].view.setValues(width=0.0149217, 
    height=0.0133948, viewOffsetX=-0.000846433, viewOffsetY=-0.00137239)
session.viewports['Viewport: 11'].view.setValues(width=0.0149698, 
    height=0.0133948, viewOffsetX=-0.000851339, viewOffsetY=-0.00137238)
session.viewports['Viewport: 12'].view.setValues(width=0.0149217, 
    height=0.0133948, viewOffsetX=-0.000846433, viewOffsetY=-0.00137239)
session.viewports['Viewport: 1'].view.setValues(width=0.0126171, 
    height=0.0464841, viewOffsetX=-0.000270517, viewOffsetY=0.00019696)
session.viewports['Viewport: 2'].view.setValues(width=0.012558, 
    height=0.052412, viewOffsetX=-7.02563e-05, viewOffsetY=0.000193931)
session.viewports['Viewport: 3'].view.setValues(width=0.0141181, 
    height=0.012628, viewOffsetX=-0.00110784, viewOffsetY=-0.00183034)
session.viewports['Viewport: 4'].view.setValues(width=0.0141312, 
    height=0.0126397, viewOffsetX=-0.00110887, viewOffsetY=-0.00183203)
session.viewports['Viewport: 10'].view.setValues(width=0.0140409, 
    height=0.0126042, viewOffsetX=-0.00110172, viewOffsetY=-0.00181356)
session.viewports['Viewport: 11'].view.setValues(width=0.0140862, 
    height=0.0126042, viewOffsetX=-0.00110808, viewOffsetY=-0.00181355)
session.viewports['Viewport: 12'].view.setValues(width=0.0140409, 
    height=0.0126042, viewOffsetX=-0.00110172, viewOffsetY=-0.00181356)
session.viewports['Viewport: 1'].view.setValues(width=0.0118802, 
    height=0.0437692, viewOffsetX=-0.000332589, viewOffsetY=0.000228021)
session.viewports['Viewport: 2'].view.setValues(width=0.0118272, 
    height=0.0493617, viewOffsetX=-8.91303e-05, viewOffsetY=0.000226344)
session.viewports['Viewport: 3'].view.setValues(width=0.013284, 
    height=0.011882, viewOffsetX=-0.00134962, viewOffsetY=-0.00224135)
session.viewports['Viewport: 4'].view.setValues(width=0.0132963, 
    height=0.011893, viewOffsetX=-0.00135087, viewOffsetY=-0.00224342)
session.viewports['Viewport: 10'].view.setValues(width=0.0132113, 
    height=0.0118595, viewOffsetX=-0.00134218, viewOffsetY=-0.00222095)
session.viewports['Viewport: 11'].view.setValues(width=0.013254, 
    height=0.0118595, viewOffsetX=-0.0013499, viewOffsetY=-0.00222095)
session.viewports['Viewport: 12'].view.setValues(width=0.0132113, 
    height=0.0118595, viewOffsetX=-0.00134218, viewOffsetY=-0.00222095)
session.viewports['Viewport: 1'].view.setValues(width=0.0111852, 
    height=0.0412087, viewOffsetX=-0.000391129, viewOffsetY=0.000257315)
session.viewports['Viewport: 2'].view.setValues(width=0.0111376, 
    height=0.0464838, viewOffsetX=-0.000106938, viewOffsetY=0.000256927)
session.viewports['Viewport: 3'].view.setValues(width=0.0124984, 
    height=0.0111793, viewOffsetX=-0.00157733, viewOffsetY=-0.00262843)
session.viewports['Viewport: 4'].view.setValues(width=0.01251, 
    height=0.0111897, viewOffsetX=-0.00157879, viewOffsetY=-0.00263087)
session.viewports['Viewport: 10'].view.setValues(width=0.0124301, 
    height=0.0111581, viewOffsetX=-0.00156864, viewOffsetY=-0.00260463)
session.viewports['Viewport: 11'].view.setValues(width=0.0124701, 
    height=0.0111581, viewOffsetX=-0.00157765, viewOffsetY=-0.00260462)
session.viewports['Viewport: 12'].view.setValues(width=0.0124301, 
    height=0.0111581, viewOffsetX=-0.00156864, viewOffsetY=-0.00260463)
session.viewports['Viewport: 1'].view.setValues(width=0.0105299, 
    height=0.0387945, viewOffsetX=-0.000446327, viewOffsetY=0.000284936)
session.viewports['Viewport: 2'].view.setValues(width=0.0104872, 
    height=0.0437691, viewOffsetX=-0.000123736, viewOffsetY=0.000285775)
session.viewports['Viewport: 3'].view.setValues(width=0.0117587, 
    height=0.0105177, viewOffsetX=-0.00179176, viewOffsetY=-0.00299295)
session.viewports['Viewport: 4'].view.setValues(width=0.0117696, 
    height=0.0105274, viewOffsetX=-0.00179342, viewOffsetY=-0.00299571)
session.viewports['Viewport: 10'].view.setValues(width=0.0116943, 
    height=0.0104977, viewOffsetX=-0.00178188, viewOffsetY=-0.00296592)
session.viewports['Viewport: 11'].view.setValues(width=0.0117321, 
    height=0.0104977, viewOffsetX=-0.00179211, viewOffsetY=-0.00296592)
session.viewports['Viewport: 12'].view.setValues(width=0.0116943, 
    height=0.0104977, viewOffsetX=-0.00178188, viewOffsetY=-0.00296592)
session.viewports['Viewport: 1'].view.setValues(width=0.00991216, 
    height=0.0365184, viewOffsetX=-0.000498365, viewOffsetY=0.000310976)
session.viewports['Viewport: 2'].view.setValues(width=0.00987373, 
    height=0.0412088, viewOffsetX=-0.000139578, viewOffsetY=0.000312982)
session.viewports['Viewport: 3'].view.setValues(width=0.0110622, 
    height=0.00989467, viewOffsetX=-0.00199366, viewOffsetY=-0.00333616)
session.viewports['Viewport: 4'].view.setValues(width=0.0110724, 
    height=0.00990383, viewOffsetX=-0.00199551, viewOffsetY=-0.00333925)
session.viewports['Viewport: 10'].view.setValues(width=0.0110016, 
    height=0.00987586, viewOffsetX=-0.00198267, viewOffsetY=-0.00330611)
session.viewports['Viewport: 11'].view.setValues(width=0.0110371, 
    height=0.00987586, viewOffsetX=-0.00199404, viewOffsetY=-0.00330611)
session.viewports['Viewport: 12'].view.setValues(width=0.0110016, 
    height=0.00987586, viewOffsetX=-0.00198267, viewOffsetY=-0.00330611)
session.viewports['Viewport: 1'].view.setValues(width=0.00932985, 
    height=0.0343731, viewOffsetX=-0.000547416, viewOffsetY=0.000335522)
session.viewports['Viewport: 2'].view.setValues(width=0.0092953, 
    height=0.0387947, viewOffsetX=-0.000154517, viewOffsetY=0.000338636)
session.viewports['Viewport: 3'].view.setValues(width=0.0104064, 
    height=0.00930812, viewOffsetX=-0.00218374, viewOffsetY=-0.0036593)
session.viewports['Viewport: 4'].view.setValues(width=0.0104161, 
    height=0.00931674, viewOffsetX=-0.00218577, viewOffsetY=-0.00366268)
session.viewports['Viewport: 10'].view.setValues(width=0.0103494, 
    height=0.0092904, viewOffsetX=-0.00217172, viewOffsetY=-0.0036264)
session.viewports['Viewport: 11'].view.setValues(width=0.0103828, 
    height=0.0092904, viewOffsetX=-0.00218415, viewOffsetY=-0.00362639)
session.viewports['Viewport: 12'].view.setValues(width=0.0103494, 
    height=0.0092904, viewOffsetX=-0.00217172, viewOffsetY=-0.0036264)
session.viewports['Viewport: 2'].setValues(width=18.3645820617676)
#* RangeError: width must be a Float in the range: 30 <= width <= 3350
session.viewports['Viewport: 1'].view.setValues(width=0.00878107, 
    height=0.0323513, viewOffsetX=-0.000600649, viewOffsetY=0.000435522)
session.viewports['Viewport: 2'].view.setValues(width=0.00874999, 
    height=0.0365188, viewOffsetX=-0.000176522, viewOffsetY=0.00044974)
session.viewports['Viewport: 3'].view.setValues(width=0.00978911, 
    height=0.00875594, viewOffsetX=-0.00236672, viewOffsetY=-0.00391927)
session.viewports['Viewport: 4'].view.setValues(width=0.00979818, 
    height=0.00876405, viewOffsetX=-0.00236892, viewOffsetY=-0.00392289)
session.viewports['Viewport: 10'].view.setValues(width=0.00973544, 
    height=0.00873925, viewOffsetX=-0.00235369, viewOffsetY=-0.00388392)
session.viewports['Viewport: 11'].view.setValues(width=0.00976684, 
    height=0.00873925, viewOffsetX=-0.00236713, viewOffsetY=-0.00388392)
session.viewports['Viewport: 12'].view.setValues(width=0.00973544, 
    height=0.00873925, viewOffsetX=-0.00235369, viewOffsetY=-0.00388392)
session.viewports['Viewport: 1'].view.setValues(width=0.00826395, 
    height=0.0304461, viewOffsetX=-0.000650811, viewOffsetY=0.000529752)
session.viewports['Viewport: 2'].view.setValues(width=0.00823598, 
    height=0.0343736, viewOffsetX=-0.000197264, viewOffsetY=0.000554466)
session.viewports['Viewport: 3'].view.setValues(width=0.00920801, 
    height=0.00823617, viewOffsetX=-0.00253897, viewOffsetY=-0.00416398)
session.viewports['Viewport: 4'].view.setValues(width=0.00921654, 
    height=0.0082438, viewOffsetX=-0.00254132, viewOffsetY=-0.00416784)
session.viewports['Viewport: 10'].view.setValues(width=0.0091575, 
    height=0.00822045, viewOffsetX=-0.00252498, viewOffsetY=-0.00412634)
session.viewports['Viewport: 11'].view.setValues(width=0.00918704, 
    height=0.00822045, viewOffsetX=-0.00253938, viewOffsetY=-0.00412633)
session.viewports['Viewport: 12'].view.setValues(width=0.0091575, 
    height=0.00822045, viewOffsetX=-0.00252498, viewOffsetY=-0.00412634)
session.viewports['Viewport: 1'].view.setValues(width=0.00777675, 
    height=0.0286511, viewOffsetX=-0.00069807, viewOffsetY=0.000618532)
session.viewports['Viewport: 2'].view.setValues(width=0.00775157, 
    height=0.0323518, viewOffsetX=-0.000216812, viewOffsetY=0.000653164)
session.viewports['Viewport: 3'].view.setValues(width=0.00866106, 
    height=0.00774694, viewOffsetX=-0.00270109, viewOffsetY=-0.00439432)
session.viewports['Viewport: 4'].view.setValues(width=0.00866908, 
    height=0.00775412, viewOffsetX=-0.00270359, viewOffsetY=-0.00439838)
session.viewports['Viewport: 10'].view.setValues(width=0.00861353, 
    height=0.00773214, viewOffsetX=-0.0026862, viewOffsetY=-0.0043545)
session.viewports['Viewport: 11'].view.setValues(width=0.00864131, 
    height=0.00773214, viewOffsetX=-0.0027015, viewOffsetY=-0.0043545)
session.viewports['Viewport: 12'].view.setValues(width=0.00861353, 
    height=0.00773214, viewOffsetX=-0.0026862, viewOffsetY=-0.0043545)
session.viewports['Viewport: 1'].view.setValues(width=0.00828179, 
    height=0.0305118, viewOffsetX=-0.000652216, viewOffsetY=0.000530896)
session.viewports['Viewport: 2'].view.setValues(width=0.00825611, 
    height=0.0344576, viewOffsetX=-0.000197746, viewOffsetY=0.000555821)
session.viewports['Viewport: 3'].view.setValues(width=0.00921942, 
    height=0.00824638, viewOffsetX=-0.00254211, viewOffsetY=-0.00416914)
session.viewports['Viewport: 4'].view.setValues(width=0.00922796, 
    height=0.00825401, viewOffsetX=-0.00254447, viewOffsetY=-0.004173)
session.viewports['Viewport: 10'].view.setValues(width=0.00916881, 
    height=0.00823061, viewOffsetX=-0.0025281, viewOffsetY=-0.00413143)
session.viewports['Viewport: 11'].view.setValues(width=0.00919838, 
    height=0.0082306, viewOffsetX=-0.00254251, viewOffsetY=-0.00413143)
session.viewports['Viewport: 12'].view.setValues(width=0.00916881, 
    height=0.00823061, viewOffsetX=-0.0025281, viewOffsetY=-0.00413143)
session.viewports['Viewport: 1'].view.setValues(width=0.00880087, 
    height=0.0324242, viewOffsetX=-0.000602004, viewOffsetY=0.000436504)
session.viewports['Viewport: 2'].view.setValues(width=0.00877228, 
    height=0.0366118, viewOffsetX=-0.000176972, viewOffsetY=0.000450885)
session.viewports['Viewport: 3'].view.setValues(width=0.00980189, 
    height=0.00876737, viewOffsetX=-0.00236981, viewOffsetY=-0.00392438)
session.viewports['Viewport: 4'].view.setValues(width=0.00981096, 
    height=0.00877549, viewOffsetX=-0.00237201, viewOffsetY=-0.00392801)
session.viewports['Viewport: 10'].view.setValues(width=0.0097481, 
    height=0.00875062, viewOffsetX=-0.00235675, viewOffsetY=-0.00388897)
session.viewports['Viewport: 11'].view.setValues(width=0.00977954, 
    height=0.00875062, viewOffsetX=-0.00237021, viewOffsetY=-0.00388897)
session.viewports['Viewport: 12'].view.setValues(width=0.0097481, 
    height=0.00875062, viewOffsetX=-0.00235675, viewOffsetY=-0.00388897)
session.viewports['Viewport: 1'].view.setValues(width=0.0093522, 
    height=0.0344554, viewOffsetX=-0.000548727, viewOffsetY=0.000336325)
session.viewports['Viewport: 2'].view.setValues(width=0.00932043, 
    height=0.0388996, viewOffsetX=-0.000154935, viewOffsetY=0.000339551)
session.viewports['Viewport: 3'].view.setValues(width=0.0104209, 
    height=0.00932102, viewOffsetX=-0.00218677, viewOffsetY=-0.00366437)
session.viewports['Viewport: 4'].view.setValues(width=0.0104305, 
    height=0.00932965, viewOffsetX=-0.0021888, viewOffsetY=-0.00366776)
session.viewports['Viewport: 10'].view.setValues(width=0.0103637, 
    height=0.00930324, viewOffsetX=-0.00217472, viewOffsetY=-0.00363141)
session.viewports['Viewport: 11'].view.setValues(width=0.0103971, 
    height=0.00930323, viewOffsetX=-0.00218717, viewOffsetY=-0.0036314)
session.viewports['Viewport: 12'].view.setValues(width=0.0103637, 
    height=0.00930324, viewOffsetX=-0.00217472, viewOffsetY=-0.00363141)
session.viewports['Viewport: 1'].view.setValues(width=0.00993735, 
    height=0.0366112, viewOffsetX=-0.000492179, viewOffsetY=0.000229995)
session.viewports['Viewport: 2'].view.setValues(width=0.00990205, 
    height=0.041327, viewOffsetX=-0.000131552, viewOffsetY=0.000221419)
session.viewports['Viewport: 3'].view.setValues(width=0.0110785, 
    height=0.00990924, viewOffsetX=-0.00199231, viewOffsetY=-0.00338813)
session.viewports['Viewport: 4'].view.setValues(width=0.0110888, 
    height=0.00991841, viewOffsetX=-0.00199415, viewOffsetY=-0.00339126)
session.viewports['Viewport: 10'].view.setValues(width=0.0110178, 
    height=0.00989036, viewOffsetX=-0.00198132, viewOffsetY=-0.00335776)
session.viewports['Viewport: 11'].view.setValues(width=0.0110533, 
    height=0.00989035, viewOffsetX=-0.0019927, viewOffsetY=-0.00335775)
session.viewports['Viewport: 12'].view.setValues(width=0.0110178, 
    height=0.00989036, viewOffsetX=-0.00198132, viewOffsetY=-0.00335776)
session.viewports['Viewport: 1'].view.setValues(width=0.0105583, 
    height=0.0388991, viewOffsetX=-0.000432169, viewOffsetY=0.000117155)
session.viewports['Viewport: 2'].view.setValues(width=0.0105191, 
    height=0.0439023, viewOffsetX=-0.000106745, viewOffsetY=9.60936e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.0117771, 
    height=0.0105341, viewOffsetX=-0.00178572, viewOffsetY=-0.00309467)
session.viewports['Viewport: 4'].view.setValues(width=0.011788, 
    height=0.0105439, viewOffsetX=-0.00178737, viewOffsetY=-0.00309753)
session.viewports['Viewport: 10'].view.setValues(width=0.0117126, 
    height=0.0105141, viewOffsetX=-0.00177587, viewOffsetY=-0.00306705)
session.viewports['Viewport: 11'].view.setValues(width=0.0117503, 
    height=0.0105141, viewOffsetX=-0.00178611, viewOffsetY=-0.00306705)
session.viewports['Viewport: 12'].view.setValues(width=0.0117126, 
    height=0.0105141, viewOffsetX=-0.00177587, viewOffsetY=-0.00306705)
session.viewports['Viewport: 1'].view.setValues(width=0.0112172, 
    height=0.0413266, viewOffsetX=-0.000389526, viewOffsetY=3.0988e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0111736, 
    height=0.0466338, viewOffsetX=-0.000104206, viewOffsetY=1.09978e-06)
session.viewports['Viewport: 3'].view.setValues(width=0.0125192, 
    height=0.0111979, viewOffsetX=-0.00157838, viewOffsetY=-0.0027636)
session.viewports['Viewport: 4'].view.setValues(width=0.0125308, 
    height=0.0112083, viewOffsetX=-0.00157985, viewOffsetY=-0.00276616)
session.viewports['Viewport: 10'].view.setValues(width=0.0124506, 
    height=0.0111766, viewOffsetX=-0.00156968, viewOffsetY=-0.00273902)
session.viewports['Viewport: 11'].view.setValues(width=0.0124908, 
    height=0.0111766, viewOffsetX=-0.0015787, viewOffsetY=-0.00273901)
session.viewports['Viewport: 12'].view.setValues(width=0.0124506, 
    height=0.0111766, viewOffsetX=-0.00156968, viewOffsetY=-0.00273902)
session.viewports['Viewport: 1'].view.setValues(width=0.0119163, 
    height=0.043902, viewOffsetX=-0.000339817, viewOffsetY=4.65434e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0118676, 
    height=0.0495306, viewOffsetX=-9.64632e-05, viewOffsetY=2.12461e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.0133074, 
    height=0.0119029, viewOffsetX=-0.00135559, viewOffsetY=-0.00235031)
session.viewports['Viewport: 4'].view.setValues(width=0.0133198, 
    height=0.0119139, viewOffsetX=-0.00135684, viewOffsetY=-0.00235248)
session.viewports['Viewport: 10'].view.setValues(width=0.0132346, 
    height=0.0118803, viewOffsetX=-0.00134811, viewOffsetY=-0.00232928)
session.viewports['Viewport: 11'].view.setValues(width=0.0132773, 
    height=0.0118803, viewOffsetX=-0.00135584, viewOffsetY=-0.00232927)
session.viewports['Viewport: 12'].view.setValues(width=0.0132346, 
    height=0.0118803, viewOffsetX=-0.00134811, viewOffsetY=-0.00232928)
session.viewports['Viewport: 1'].view.setValues(width=0.0126577, 
    height=0.0466337, viewOffsetX=-0.000277597, viewOffsetY=6.30433e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0126036, 
    height=0.0526021, viewOffsetX=-7.7527e-05, viewOffsetY=4.26071e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.0141446, 
    height=0.0126517, viewOffsetX=-0.0011135, viewOffsetY=-0.00191137)
session.viewports['Viewport: 4'].view.setValues(width=0.0141576, 
    height=0.0126634, viewOffsetX=-0.00111453, viewOffsetY=-0.00191314)
session.viewports['Viewport: 10'].view.setValues(width=0.0140671, 
    height=0.0126277, viewOffsetX=-0.00110734, viewOffsetY=-0.00189412)
session.viewports['Viewport: 11'].view.setValues(width=0.0141125, 
    height=0.0126277, viewOffsetX=-0.00111371, viewOffsetY=-0.00189411)
session.viewports['Viewport: 12'].view.setValues(width=0.0140671, 
    height=0.0126277, viewOffsetX=-0.00110734, viewOffsetY=-0.00189412)
session.viewports['Viewport: 10'].setValues(height=138.942123413086)
session.viewports['Viewport: 10'].setValues(origin=(72.3385391235352, 
    11.6157379150391), width=79.5052032470703)
session.viewports['Viewport: 10'].setValues(width=22.84375)
#* RangeError: width must be a Float in the range: 30 <= width <= 3350
session.viewports['Viewport: 10'].setValues(width=45.2395820617676)
session.viewports['Viewport: 10'].setValues(width=30.6822910308838)
session.viewports['Viewport: 10'].setValues(width=36.28125)
session.viewports['Viewport: 1'].setValues(width=36.7291641235352)
session.viewports['Viewport: 2'].setValues(origin=(36.28125, 11.839111328125))
session.viewports['Viewport: 10'].setValues(origin=(70.0989532470703, 
    12.285888671875))
session.viewports['Viewport: 10'].setValues(origin=(105.484375, 
    10.9456024169922))
session.viewports['Viewport: 2'].setValues(origin=(63.3802070617676, 
    8.26504516601562))
session.viewports['Viewport: 1'].setValues(origin=(157.44270324707, 
    -0.670135498046875))
session.viewports['Viewport: 10'].setValues(origin=(-0.447916656732559, 
    12.0625))
o3 = session.odbs['C:/LocalUserData/User-data/nguyenb5/CP1000 plastic (UMAT UMATHT)/CP1000_dense_CSC/CP1000_CHD2_dense_CSC/CHD2_combined_processed.odb']
session.viewports['Viewport: 11'].setValues(displayedObject=o3)
session.viewports['Viewport: 10'].makeCurrent()
o3 = session.odbs['C:/LocalUserData/User-data/nguyenb5/CP1000 plastic (UMAT UMATHT)/CP1000_dense_CSC/CP1000_CHD2_dense_CSC/CHD2_combined_processed.odb']
session.viewports['Viewport: 10'].setValues(displayedObject=o3)
session.viewports['Viewport: 10'].view.setValues(session.views['Front'])
session.viewports['Viewport: 10'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM1', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 10'].odbDisplay.display.setValues(
    plotState=CONTOURS_ON_DEF)
session.viewports['Viewport: 2'].setValues(origin=(103.02082824707, 
    2.23379516601562))
session.viewports['Viewport: 1'].setValues(origin=(34.9375, 11.1689910888672))
session.viewports['Viewport: 2'].setValues(origin=(71.890625, 
    11.3923645019531))
session.viewports['Viewport: 10'].setValues(origin=(-0.671875, 
    10.4988403320312))
session.viewports['Viewport: 10'].setValues(origin=(-1.11979162693024, 
    11.6157379150391))
session.viewports['Viewport: 10'].setValues(origin=(-0.447916656732559, 
    11.1689910888672))
session.viewports['Viewport: 11'].setValues(origin=(105.70832824707, 
    81.9803161621094))
session.viewports['Viewport: 11'].setValues(width=33.59375)
session.viewports['Viewport: 11'].setValues(origin=(105.70832824707, 
    11.6157379150391), height=139.388885498047)
session.viewports['Viewport: 3'].setValues(origin=(179.61457824707, 
    81.3101806640625), width=35.3854141235352)
session.viewports['Viewport: 12'].setValues(origin=(139.078125, 
    81.3101806640625))
session.viewports['Viewport: 12'].setValues(width=36.7291641235352)
session.viewports['Viewport: 12'].setValues(origin=(139.078125, 
    11.6157379150391), height=138.71875)
session.viewports['Viewport: 3'].setValues(origin=(175.359375, 
    81.3101806640625), width=39.4166641235352)
session.viewports['Viewport: 3'].setValues(origin=(175.359375, 
    11.8391265869141), height=138.495361328125)
session.viewports['Viewport: 3'].setValues(origin=(175.359375, 
    10.9456024169922), height=139.165512084961)
session.viewports['Viewport: 3'].makeCurrent()
session.viewports['Viewport: 4'].setValues(origin=(215.0, 10.9456024169922), 
    height=139.388885498047)
session.viewports['Viewport: 4'].setValues(width=36.5052070617676)
session.viewports['Viewport: 3'].view.setValues(session.views['Back'])
session.viewports['Viewport: 3'].view.setValues(session.views['Back'])
session.viewports['Viewport: 3'].view.setValues(session.views['Front'])
session.viewports['Viewport: 3'].view.setValues(session.views['Front'])
session.viewports['Viewport: 3'].view.setValues(session.views['Back'])
session.viewports['Viewport: 3'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM2', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 3'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM1', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].view.setValues(width=0.0637301, 
    height=0.243943, viewOffsetX=-5.28297e-05, viewOffsetY=-3.95206e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0583029, 
    height=0.243332, viewOffsetX=0.00011877, viewOffsetY=-5.265e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.0687635, 
    height=0.244595, viewOffsetX=-0.000211526, viewOffsetY=-2.63724e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0632458, 
    height=0.244496, viewOffsetX=-3.95783e-05, viewOffsetY=-1.31589e-05)
session.viewports['Viewport: 10'].view.setValues(width=0.062961, 
    height=0.24417, viewOffsetX=-2.64394e-05, viewOffsetY=-3.95572e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.0577065, 
    height=0.243797, viewOffsetX=0.00013155, viewOffsetY=-1.31214e-05)
session.viewports['Viewport: 12'].view.setValues(width=0.0634163, 
    height=0.243915, viewOffsetX=-3.96849e-05, viewOffsetY=-5.27769e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.0524406, 
    height=0.200729, viewOffsetY=-6.7115e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0480513, 
    height=0.200546, viewOffsetX=0.000245526, viewOffsetY=-8.95553e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.0565516, 
    height=0.201157, viewOffsetX=-0.000315534, viewOffsetY=-4.47625e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0521029, 
    height=0.201419, viewOffsetX=-2.38181e-05, viewOffsetY=-2.23731e-05)
session.viewports['Viewport: 10'].view.setValues(width=0.0518189, 
    height=0.200959, viewOffsetX=-1.38918e-06, viewOffsetY=-6.71916e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.0476048, 
    height=0.201119, viewOffsetX=0.000267379, viewOffsetY=-2.23398e-05)
session.viewports['Viewport: 12'].view.setValues(width=0.052175, 
    height=0.200678, viewOffsetX=-2.38509e-05, viewOffsetY=-8.96146e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.0507131, 
    height=0.194117, viewOffsetY=-0.000100495)
session.viewports['Viewport: 2'].view.setValues(width=0.0464589, height=0.1939, 
    viewOffsetX=0.000389247, viewOffsetY=-0.000134069)
session.viewports['Viewport: 3'].view.setValues(width=0.0546906, 
    height=0.194537, viewOffsetX=-0.000450805, viewOffsetY=-6.70279e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0503724, 
    height=0.19473, viewOffsetX=-1.39895e-05, viewOffsetY=-3.34914e-05)
session.viewports['Viewport: 10'].view.setValues(width=0.0501102, 
    height=0.194333, viewOffsetX=1.96136e-05, viewOffsetY=-0.000100607)
session.viewports['Viewport: 11'].view.setValues(width=0.0460139, 
    height=0.194398, viewOffsetX=0.000421792, viewOffsetY=-3.34344e-05)
session.viewports['Viewport: 12'].view.setValues(width=0.0504597, 
    height=0.194081, viewOffsetX=-1.40136e-05, viewOffsetY=-0.000134194)
session.viewports['Viewport: 1'].view.setValues(width=0.0478744, 
    height=0.183251, viewOffsetY=-9.103e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0438598, 
    height=0.183052, viewOffsetX=0.000519984)
session.viewports['Viewport: 3'].view.setValues(width=0.0516286, 
    height=0.183646, viewOffsetX=-0.000571842, viewOffsetY=-4.75131e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0475538, 
    height=0.183834, viewOffsetX=-4.1303e-06, viewOffsetY=-3.95463e-06)
session.viewports['Viewport: 10'].view.setValues(width=0.0473054, 
    height=0.183456, viewOffsetX=3.95626e-05, viewOffsetY=-9.11311e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.0434403, 
    height=0.183525, viewOffsetX=0.000562258, viewOffsetY=-3.94747e-06)
session.viewports['Viewport: 12'].view.setValues(width=0.0476351, 
    height=0.183216, viewOffsetX=-4.13724e-06)
session.viewports['Viewport: 1'].view.setValues(width=0.0453172, 
    height=0.173463, viewOffsetX=-6.68986e-05, viewOffsetY=-8.23005e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0415173, 
    height=0.173276, viewOffsetX=0.000627001)
session.viewports['Viewport: 3'].view.setValues(width=0.0488703, 
    height=0.173834, viewOffsetX=-0.000707382, viewOffsetY=-2.91025e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0450125, 
    height=0.174009, viewOffsetX=-1.35485e-05, viewOffsetY=2.4113e-05)
session.viewports['Viewport: 10'].view.setValues(width=0.0447786, 
    height=0.173656, viewOffsetX=3.98395e-05, viewOffsetY=-8.2392e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.0411188, 
    height=0.173717, viewOffsetX=0.000678663, viewOffsetY=2.40729e-05)
session.viewports['Viewport: 12'].view.setValues(width=0.0450912, 
    height=0.173432, viewOffsetX=-1.35721e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.0428651, 
    height=0.164077, viewOffsetX=-0.000102851, viewOffsetY=-7.3956e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0392712, 
    height=0.163902, viewOffsetX=0.000710937)
session.viewports['Viewport: 3'].view.setValues(width=0.0462255, 
    height=0.164427, viewOffsetX=-0.000854003, viewOffsetY=-1.15561e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0425761, 
    height=0.164591, viewOffsetX=-4.02767e-05, viewOffsetY=5.08376e-05)
session.viewports['Viewport: 10'].view.setValues(width=0.0423556, 
    height=0.16426, viewOffsetX=2.23026e-05, viewOffsetY=-7.40382e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.0388932, 
    height=0.164315, viewOffsetX=0.000771567, viewOffsetY=5.07531e-05)
session.viewports['Viewport: 12'].view.setValues(width=0.0426518, 
    height=0.164049, viewOffsetX=-4.03481e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.0405338, 
    height=0.155153, viewOffsetX=-0.000137066, viewOffsetY=-6.60193e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0371357, 
    height=0.154989, viewOffsetX=0.000790839)
session.viewports['Viewport: 3'].view.setValues(width=0.043711, 
    height=0.155482, viewOffsetX=-0.000993556, viewOffsetY=5.13924e-06)
session.viewports['Viewport: 4'].view.setValues(width=0.0402598, 
    height=0.155637, viewOffsetX=-6.57104e-05, viewOffsetY=7.62686e-05)
session.viewports['Viewport: 10'].view.setValues(width=0.0400521, 
    height=0.155326, viewOffsetX=5.61649e-06, viewOffsetY=-6.60927e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.0367772, 
    height=0.155375, viewOffsetX=0.000859999, viewOffsetY=7.61419e-05)
session.viewports['Viewport: 12'].view.setValues(width=0.0403325, 
    height=0.155129, viewOffsetX=-6.58288e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.0383168, 
    height=0.146667, viewOffsetY=-0.000121834)
session.viewports['Viewport: 2'].view.setValues(width=0.0351049, 
    height=0.146513, viewOffsetX=0.000898606, viewOffsetY=-0.000201437)
session.viewports['Viewport: 3'].view.setValues(width=0.0413199, 
    height=0.146977, viewOffsetX=-0.00109448, viewOffsetY=-4.23731e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0380572, 
    height=0.147121, viewOffsetX=-5.81416e-05, viewOffsetY=3.71078e-05)
session.viewports['Viewport: 10'].view.setValues(width=0.0378614, 
    height=0.146831, viewOffsetX=2.15473e-05, viewOffsetY=-0.000121969)
session.viewports['Viewport: 11'].view.setValues(width=0.0347651, 
    height=0.146875, viewOffsetX=0.00097579, viewOffsetY=3.70467e-05)
session.viewports['Viewport: 12'].view.setValues(width=0.0381268, 
    height=0.146645, viewOffsetX=-5.82478e-05, viewOffsetY=-0.000201619)
session.viewports['Viewport: 1'].view.setValues(width=0.03621, height=0.138603, 
    viewOffsetY=-0.000174877)
session.viewports['Viewport: 2'].view.setValues(width=0.033175, 
    height=0.138459, viewOffsetX=0.00100103, viewOffsetY=-0.000262431)
session.viewports['Viewport: 3'].view.setValues(width=0.0390477, 
    height=0.138894, viewOffsetX=-0.0011904, viewOffsetY=-8.75258e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0359641, 
    height=0.13903, viewOffsetX=-5.09491e-05, viewOffsetY=-1.0751e-07)
session.viewports['Viewport: 10'].view.setValues(width=0.0357797, 
    height=0.138758, viewOffsetX=3.66872e-05, viewOffsetY=-0.000175072)
session.viewports['Viewport: 11'].view.setValues(width=0.0328531, 
    height=0.138797, viewOffsetX=0.00108583, viewOffsetY=-1.05996e-07)
session.viewports['Viewport: 12'].view.setValues(width=0.0360308, 
    height=0.138583, viewOffsetX=-5.10434e-05, viewOffsetY=-0.000262668)
session.viewports['Viewport: 1'].view.setValues(width=0.0342091, 
    height=0.130944, viewOffsetY=-0.000225258)
session.viewports['Viewport: 2'].view.setValues(width=0.0313421, 
    height=0.130809, viewOffsetX=0.00109831, viewOffsetY=-0.000320364)
session.viewports['Viewport: 3'].view.setValues(width=0.0368897, 
    height=0.131218, viewOffsetX=-0.00128151, viewOffsetY=-0.000130411)
session.viewports['Viewport: 4'].view.setValues(width=0.0339762, 
    height=0.131345, viewOffsetX=-4.4118e-05, viewOffsetY=-3.54528e-05)
session.viewports['Viewport: 10'].view.setValues(width=0.0338026, 
    height=0.13109, viewOffsetX=5.10671e-05, viewOffsetY=-0.000225509)
session.viewports['Viewport: 11'].view.setValues(width=0.0310372, 
    height=0.131125, viewOffsetX=0.00119035, viewOffsetY=-3.53919e-05)
session.viewports['Viewport: 12'].view.setValues(width=0.03404, 
    height=0.130926, viewOffsetX=-4.42008e-05, viewOffsetY=-0.000320652)
session.viewports['Viewport: 1'].view.setValues(width=0.0323099, 
    height=0.123674, viewOffsetX=-0.00015333, viewOffsetY=-0.000206296)
session.viewports['Viewport: 2'].view.setValues(width=0.0296022, 
    height=0.123547, viewOffsetX=0.00117726, viewOffsetY=-0.000308527)
session.viewports['Viewport: 3'].view.setValues(width=0.0348413, 
    height=0.123932, viewOffsetX=-0.00138138, viewOffsetY=-0.000104306)
session.viewports['Viewport: 4'].view.setValues(width=0.0320894, 
    height=0.124051, viewOffsetX=-5.10214e-05, viewOffsetY=-2.23942e-06)
session.viewports['Viewport: 10'].view.setValues(width=0.0319259, 
    height=0.123812, viewOffsetX=5.13104e-05, viewOffsetY=-0.000206526)
session.viewports['Viewport: 11'].view.setValues(width=0.0293136, 
    height=0.123843, viewOffsetX=0.00127619, viewOffsetY=-2.23413e-06)
session.viewports['Viewport: 12'].view.setValues(width=0.0321504, 
    height=0.123658, viewOffsetX=-5.11183e-05, viewOffsetY=-0.000308805)
session.viewports['Viewport: 1'].view.setValues(width=0.0305081, 
    height=0.116777, viewOffsetX=-0.000166666, viewOffsetY=-0.000188307)
session.viewports['Viewport: 2'].view.setValues(width=0.0279516, 
    height=0.116658, viewOffsetX=0.00125216, viewOffsetY=-0.000297298)
session.viewports['Viewport: 3'].view.setValues(width=0.0328981, 
    height=0.11702, viewOffsetX=-0.00147614, viewOffsetY=-7.95409e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0302995, 
    height=0.117132, viewOffsetX=-5.75708e-05, viewOffsetY=2.92702e-05)
session.viewports['Viewport: 10'].view.setValues(width=0.0301455, 
    height=0.116908, viewOffsetX=5.15414e-05, viewOffsetY=-0.000188517)
session.viewports['Viewport: 11'].view.setValues(width=0.0276785, 
    height=0.116935, viewOffsetX=0.00135764, viewOffsetY=2.92229e-05)
session.viewports['Viewport: 12'].view.setValues(width=0.0303577, 
    height=0.116763, viewOffsetX=-5.76814e-05, viewOffsetY=-0.000297566)
session.viewports['Viewport: 1'].view.setValues(width=0.0287996, 
    height=0.110237, viewOffsetX=-0.000179311, viewOffsetY=-0.000171249)
session.viewports['Viewport: 2'].view.setValues(width=0.0263864, 
    height=0.110126, viewOffsetX=0.00132319, viewOffsetY=-0.00028665)
session.viewports['Viewport: 3'].view.setValues(width=0.0310555, 
    height=0.110466, viewOffsetX=-0.001566, viewOffsetY=-5.60575e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0286023, 
    height=0.110571, viewOffsetX=-6.37812e-05, viewOffsetY=5.91485e-05)
session.viewports['Viewport: 10'].view.setValues(width=0.0284573, 
    height=0.110361, viewOffsetX=5.17605e-05, viewOffsetY=-0.00017144)
session.viewports['Viewport: 11'].view.setValues(width=0.0261281, 
    height=0.110385, viewOffsetX=0.00143487, viewOffsetY=5.90511e-05)
session.viewports['Viewport: 12'].view.setValues(width=0.0286578, 
    height=0.110225, viewOffsetX=-6.3905e-05, viewOffsetY=-0.000286909)
session.viewports['Viewport: 1'].view.setValues(width=0.0271803, 
    height=0.104039, viewOffsetX=-0.000191297, viewOffsetY=-0.000155082)
session.viewports['Viewport: 2'].view.setValues(width=0.024903, 
    height=0.103935, viewOffsetX=0.00139052, viewOffsetY=-0.000276559)
session.viewports['Viewport: 3'].view.setValues(width=0.0293092, 
    height=0.104254, viewOffsetX=-0.00165117, viewOffsetY=-3.38006e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0269938, 
    height=0.104353, viewOffsetX=-6.96673e-05, viewOffsetY=8.74663e-05)
session.viewports['Viewport: 10'].view.setValues(width=0.0268573, 
    height=0.104156, viewOffsetX=5.19682e-05, viewOffsetY=-0.000155254)
session.viewports['Viewport: 11'].view.setValues(width=0.0246588, 
    height=0.104178, viewOffsetX=0.00150807, viewOffsetY=8.73213e-05)
session.viewports['Viewport: 12'].view.setValues(width=0.0270467, 
    height=0.104028, viewOffsetX=-6.98038e-05, viewOffsetY=-0.000276808)
session.viewports['Viewport: 1'].view.setValues(width=0.0256463, 
    height=0.0981677, viewOffsetX=-0.000202652, viewOffsetY=-0.000129164)
session.viewports['Viewport: 2'].view.setValues(width=0.0234977, 
    height=0.0980696, viewOffsetX=0.0014543, viewOffsetY=-0.000256389)
session.viewports['Viewport: 3'].view.setValues(width=0.0276549, 
    height=0.0983699, viewOffsetX=-0.00173185, viewOffsetY=-2.1095e-06)
session.viewports['Viewport: 4'].view.setValues(width=0.02547, 
    height=0.0984621, viewOffsetX=-7.52436e-05, viewOffsetY=0.000124891)
session.viewports['Viewport: 10'].view.setValues(width=0.0253416, 
    height=0.0982773, viewOffsetX=5.21651e-05, viewOffsetY=-0.000129307)
session.viewports['Viewport: 11'].view.setValues(width=0.0232668, 
    height=0.098297, viewOffsetX=0.00157742, viewOffsetY=0.000124683)
session.viewports['Viewport: 12'].view.setValues(width=0.0255204, 
    height=0.0981578, viewOffsetX=-7.53923e-05, viewOffsetY=-0.000256621)
session.viewports['Viewport: 1'].view.setValues(width=0.0241938, 
    height=0.0926078, viewOffsetY=-0.000144629)
session.viewports['Viewport: 2'].view.setValues(width=0.022167, 
    height=0.0925158, viewOffsetX=0.00152474, viewOffsetY=-0.000277326)
session.viewports['Viewport: 3'].view.setValues(width=0.0260885, 
    height=0.092798, viewOffsetX=-0.00179823, viewOffsetY=-1.21226e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0240272, 
    height=0.0928844, viewOffsetX=-7.05e-05, viewOffsetY=0.000120337)
session.viewports['Viewport: 10'].view.setValues(width=0.0239063, 
    height=0.0927112, viewOffsetX=6.23908e-05, viewOffsetY=-0.00014479)
session.viewports['Viewport: 11'].view.setValues(width=0.0219488, 
    height=0.0927287, viewOffsetX=0.00165309, viewOffsetY=0.000120137)
session.viewports['Viewport: 12'].view.setValues(width=0.0240752, 
    height=0.0925989, viewOffsetX=-7.06406e-05, viewOffsetY=-0.000277576)
session.viewports['Viewport: 1'].view.setValues(width=0.022819, 
    height=0.0873452, viewOffsetY=-0.000159267)
session.viewports['Viewport: 2'].view.setValues(width=0.0209074, 
    height=0.0872589, viewOffsetX=0.00159142, viewOffsetY=-0.000297144)
session.viewports['Viewport: 3'].view.setValues(width=0.0246058, 
    height=0.0875241, viewOffsetX=-0.00186106, viewOffsetY=-2.16003e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0226616, 
    height=0.0876052, viewOffsetX=-6.60103e-05, viewOffsetY=0.000116027)
session.viewports['Viewport: 10'].view.setValues(width=0.0225478, 
    height=0.0874427, viewOffsetX=7.20699e-05, viewOffsetY=-0.000159445)
session.viewports['Viewport: 11'].view.setValues(width=0.0207013, 
    height=0.0874583, viewOffsetX=0.00172472, viewOffsetY=0.000115834)
session.viewports['Viewport: 12'].view.setValues(width=0.0227072, 
    height=0.0873373, viewOffsetX=-6.61429e-05, viewOffsetY=-0.000297412)
session.viewports['Viewport: 1'].view.setValues(width=0.0215181, 
    height=0.0823659, viewOffsetY=-0.000173119)
session.viewports['Viewport: 2'].view.setValues(width=0.0197157, 
    height=0.0822849, viewOffsetX=0.00165451, viewOffsetY=-0.000315896)
session.viewports['Viewport: 3'].view.setValues(width=0.023203, 
    height=0.0825342, viewOffsetX=-0.0019205, viewOffsetY=-3.05679e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0213695, 
    height=0.0826102, viewOffsetX=-6.17622e-05, viewOffsetY=0.000111949)
session.viewports['Viewport: 10'].view.setValues(width=0.0212624, 
    height=0.0824579, viewOffsetX=8.12281e-05, viewOffsetY=-0.000173311)
session.viewports['Viewport: 11'].view.setValues(width=0.019521, 
    height=0.0824718, viewOffsetX=0.0017925, viewOffsetY=0.000111763)
session.viewports['Viewport: 12'].view.setValues(width=0.0214128, 
    height=0.0823589, viewOffsetX=-6.18873e-05, viewOffsetY=-0.000316181)
session.viewports['Viewport: 1'].view.setValues(width=0.0202878, 
    height=0.0776565, viewOffsetY=-0.000186219)
session.viewports['Viewport: 2'].view.setValues(width=0.0185885, 
    height=0.0775805, viewOffsetX=0.00171418, viewOffsetY=-0.000333633)
session.viewports['Viewport: 3'].view.setValues(width=0.0218762, 
    height=0.0778147, viewOffsetX=-0.00197673, viewOffsetY=-3.90495e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0201475, 
    height=0.0778861, viewOffsetX=-5.77445e-05, viewOffsetY=0.000108092)
session.viewports['Viewport: 10'].view.setValues(width=0.0200467, 
    height=0.0777432, viewOffsetX=8.98903e-05, viewOffsetY=-0.000186427)
session.viewports['Viewport: 11'].view.setValues(width=0.0184047, 
    height=0.0777556, viewOffsetX=0.00185661, viewOffsetY=0.000107912)
session.viewports['Viewport: 12'].view.setValues(width=0.0201886, 
    height=0.0776503, viewOffsetX=-5.78623e-05, viewOffsetY=-0.000333934)
session.viewports['Viewport: 1'].view.setValues(width=0.0191245, 
    height=0.0732038, viewOffsetY=-0.000198606)
session.viewports['Viewport: 2'].view.setValues(width=0.0175227, 
    height=0.0731326, viewOffsetX=0.0017706, viewOffsetY=-0.000350403)
session.viewports['Viewport: 3'].view.setValues(width=0.0206218, 
    height=0.0733527, viewOffsetX=-0.0020299, viewOffsetY=-4.70688e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0189921, 
    height=0.0734196, viewOffsetX=-5.39459e-05, viewOffsetY=0.000104446)
session.viewports['Viewport: 10'].view.setValues(width=0.0188972, 
    height=0.0732856, viewOffsetX=9.80804e-05, viewOffsetY=-0.000198828)
session.viewports['Viewport: 11'].view.setValues(width=0.0173493, 
    height=0.0732966, viewOffsetX=0.00191722, viewOffsetY=0.000104272)
session.viewports['Viewport: 12'].view.setValues(width=0.0190311, 
    height=0.0731983, viewOffsetX=-5.40567e-05, viewOffsetY=-0.000350719)
session.viewports['Viewport: 1'].view.setValues(width=0.0180251, 
    height=0.0689953, viewOffsetY=-0.000210314)
session.viewports['Viewport: 2'].view.setValues(width=0.0165154, 
    height=0.0689285, viewOffsetX=0.00182393, viewOffsetY=-0.000366255)
session.viewports['Viewport: 3'].view.setValues(width=0.0194361, 
    height=0.0691353, viewOffsetX=-0.00208015, viewOffsetY=-5.46484e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0179001, 
    height=0.0691981, viewOffsetX=-5.03556e-05, viewOffsetY=0.000101)
session.viewports['Viewport: 10'].view.setValues(width=0.0178108, 
    height=0.0690724, viewOffsetX=0.000105822, viewOffsetY=-0.000210549)
session.viewports['Viewport: 11'].view.setValues(width=0.0163517, 
    height=0.0690822, viewOffsetX=0.00197451, viewOffsetY=0.000100832)
session.viewports['Viewport: 12'].view.setValues(width=0.0179371, 
    height=0.0689904, viewOffsetX=-5.04597e-05, viewOffsetY=-0.000366585)
session.viewports['Viewport: 4'].setValues(origin=(214.776031494141, 
    10.9456024169922), width=36.7291641235352)
session.viewports['Viewport: 4'].makeCurrent()
session.viewports['Viewport: 3'].setValues(width=32.4739570617676)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.483705, 
    farPlane=0.524131, width=0.0180686, height=0.0691618, 
    viewOffsetX=0.000596088, viewOffsetY=-0.00102861)
session.viewports['Viewport: 2'].view.setValues(nearPlane=0.483237, 
    farPlane=0.52369, width=0.0165553, height=0.0690951, 
    viewOffsetX=0.00256414, viewOffsetY=-0.00118414)
session.viewports['Viewport: 3'].view.setValues(nearPlane=0.480391, 
    farPlane=0.529536, width=0.019308, height=0.0844502, 
    viewOffsetX=-0.0012083, viewOffsetY=-0.00105285)
session.viewports['Viewport: 4'].view.setValues(nearPlane=0.485139, 
    farPlane=0.525599, width=0.0180602, height=0.0693638, 
    viewOffsetX=0.000752199, viewOffsetY=-0.000718934)
session.viewports['Viewport: 10'].view.setValues(nearPlane=0.484241, 
    farPlane=0.524712, width=0.0178538, height=0.0692391, 
    viewOffsetX=0.00089958, viewOffsetY=-0.00102976)
session.viewports['Viewport: 11'].view.setValues(nearPlane=0.484306, 
    farPlane=0.524647, width=0.016391, height=0.0692484, 
    viewOffsetX=0.00270775, viewOffsetY=-0.000717735)
session.viewports['Viewport: 12'].view.setValues(nearPlane=0.483673, 
    farPlane=0.524163, width=0.0179804, height=0.0691572, 
    viewOffsetX=0.000748549, viewOffsetY=-0.00118521)
session.viewports['Viewport: 3'].setValues(width=30.234375)
session.viewports['Viewport: 3'].setValues(width=32.6979141235352)
session.viewports['Viewport: 11'].makeCurrent()
o3 = session.odbs['C:/LocalUserData/User-data/nguyenb5/CP1000 plastic (UMAT UMATHT)/CP1000_dense_CSC/CP1000_NDBR15_dense_CSC/NDBR15_combined_processed.odb']
session.viewports['Viewport: 11'].setValues(displayedObject=o3)
session.viewports['Viewport: 12'].makeCurrent()
session.viewports['Viewport: 11'].makeCurrent()
o3 = session.odbs['C:/LocalUserData/User-data/nguyenb5/CP1000 plastic (UMAT UMATHT)/CP1000_dense_CSC/CP1000_NDBR6_dense_CSC/NDBR6_combined_processed.odb']
session.viewports['Viewport: 11'].setValues(displayedObject=o3)
session.viewports['Viewport: 12'].makeCurrent()
o3 = session.odbs['C:/LocalUserData/User-data/nguyenb5/CP1000 plastic (UMAT UMATHT)/CP1000_dense_CSC/CP1000_NDBR15_dense_CSC/NDBR15_combined_processed.odb']
session.viewports['Viewport: 12'].setValues(displayedObject=o3)
session.viewports['Viewport: 11'].makeCurrent()
session.viewports['Viewport: 12'].makeCurrent()
session.viewports['Viewport: 12'].view.setValues(session.views['Front'])
session.viewports['Viewport: 12'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM1', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 12'].odbDisplay.display.setValues(
    plotState=CONTOURS_ON_DEF)
session.viewports['Viewport: 4'].setValues(origin=(210.072906494141, 
    11.1689758300781))
session.viewports['Viewport: 12'].setValues(width=35.3854141235352)
session.viewports['Viewport: 3'].setValues(origin=(173.791656494141, 
    11.6157379150391))
session.viewports['Viewport: 3'].setValues(width=34.265625)
session.viewports['Viewport: 4'].setValues(origin=(207.609375, 
    11.3923645019531))
session.viewports['Viewport: 4'].setValues(width=33.59375)
session.viewports['Viewport: 1'].view.setValues(width=0.0637301, 
    height=0.243943, viewOffsetX=-0.000184904, viewOffsetY=3.95197e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0583029, 
    height=0.243332, viewOffsetX=-1.31966e-05, viewOffsetY=2.63259e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.0590752, 
    height=0.243931, viewOffsetX=-3.95532e-05, viewOffsetY=5.26017e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0577744, 
    height=0.244084, viewOffsetY=6.5682e-05)
session.viewports['Viewport: 10'].view.setValues(width=0.062961, 
    height=0.24417, viewOffsetX=-0.000158636, viewOffsetY=3.95564e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.0572886, 
    height=0.242031, viewOffsetY=6.51303e-05)
session.viewports['Viewport: 12'].view.setValues(width=0.0605151, 
    height=0.242256, viewOffsetX=-9.19682e-05, viewOffsetY=2.62088e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.0524406, 
    height=0.200729, viewOffsetX=-0.000292274, viewOffsetY=6.71137e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0480513, 
    height=0.200546, viewOffsetX=-6.94039e-07, viewOffsetY=4.47782e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.0487376, 
    height=0.201246, viewOffsetX=-4.55919e-05, viewOffsetY=8.95637e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0476888, 
    height=0.201474, viewOffsetX=2.17427e-05, viewOffsetY=0.000111893)
session.viewports['Viewport: 10'].view.setValues(width=0.0518189, 
    height=0.200959, viewOffsetX=-0.000247699, viewOffsetY=6.71904e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.0472673, 
    height=0.199694, viewOffsetX=2.15504e-05, viewOffsetY=0.000110904)
session.viewports['Viewport: 12'].view.setValues(width=0.0498346, 
    height=0.199499, viewOffsetX=-0.000134668, viewOffsetY=4.4544e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.0507131, 
    height=0.194117, viewOffsetX=-0.000426804, viewOffsetY=0.000100493)
session.viewports['Viewport: 2'].view.setValues(width=0.0464589, height=0.1939, 
    viewOffsetX=9.80195e-06, viewOffsetY=6.70352e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.0471103, 
    height=0.194526, viewOffsetX=-5.73966e-05, viewOffsetY=0.000134047)
session.viewports['Viewport: 4'].view.setValues(width=0.0460908, 
    height=0.194723, viewOffsetX=4.33694e-05, viewOffsetY=0.000167445)
session.viewports['Viewport: 10'].view.setValues(width=0.0501102, 
    height=0.194333, viewOffsetX=-0.000360035, viewOffsetY=0.000100604)
session.viewports['Viewport: 11'].view.setValues(width=0.0456865, 
    height=0.193015, viewOffsetX=4.29889e-05, viewOffsetY=0.000165977)
session.viewports['Viewport: 12'].view.setValues(width=0.0481891, 
    height=0.192912, viewOffsetX=-0.000190845, viewOffsetY=6.66929e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.0478744, 
    height=0.183251, viewOffsetX=-0.000527846, viewOffsetY=7.12354e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0438598, 
    height=0.183052, viewOffsetX=3.96269e-05, viewOffsetY=2.77167e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.0444756, 
    height=0.183647, viewOffsetX=-4.77192e-05, viewOffsetY=0.000114827)
session.viewports['Viewport: 4'].view.setValues(width=0.0435135, 
    height=0.183834, viewOffsetX=8.32357e-05, viewOffsetY=0.000158278)
session.viewports['Viewport: 10'].view.setValues(width=0.0473054, 
    height=0.183456, viewOffsetX=-0.000441038, viewOffsetY=7.13145e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.0431315, 
    height=0.182221, viewOffsetX=8.25049e-05, viewOffsetY=0.000156889)
session.viewports['Viewport: 12'].view.setValues(width=0.0454925, 
    height=0.182117, viewOffsetX=-0.000221296, viewOffsetY=2.75743e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.0453172, 
    height=0.173463, viewOffsetX=-0.000644242, viewOffsetY=4.36324e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0415173, 
    height=0.173276, viewOffsetX=4.93022e-05, viewOffsetY=-9.58127e-06)
session.viewports['Viewport: 3'].view.setValues(width=0.0420991, 
    height=0.173834, viewOffsetX=-5.74482e-05, viewOffsetY=9.6886e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0411879, 
    height=0.174009, viewOffsetX=0.000102595, viewOffsetY=0.000150016)
session.viewports['Viewport: 10'].view.setValues(width=0.0447786, 
    height=0.173656, viewOffsetX=-0.000538148, viewOffsetY=4.36803e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.0408264, 
    height=0.172482, viewOffsetX=0.000101694, viewOffsetY=0.0001487)
session.viewports['Viewport: 12'].view.setValues(width=0.0430629, 
    height=0.172391, viewOffsetX=-0.000269595, viewOffsetY=-9.53284e-06)
session.viewports['Viewport: 1'].view.setValues(width=0.0428651, 
    height=0.164077, viewOffsetX=-0.000754879, viewOffsetY=1.73243e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0392712, 
    height=0.163902, viewOffsetX=5.85006e-05, viewOffsetY=-4.51056e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.0398208, 
    height=0.164427, viewOffsetX=-6.66948e-05, viewOffsetY=7.97633e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0389586, 
    height=0.164591, viewOffsetX=0.000120998, viewOffsetY=0.000142095)
session.viewports['Viewport: 10'].view.setValues(width=0.0423557, 
    height=0.16426, viewOffsetX=-0.000630453, viewOffsetY=1.7343e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.0386166, 
    height=0.163146, viewOffsetX=0.000119936, viewOffsetY=0.000140848)
session.viewports['Viewport: 12'].view.setValues(width=0.0407332, 
    height=0.163065, viewOffsetX=-0.000315504, viewOffsetY=-4.48758e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.0405338, 
    height=0.155153, viewOffsetX=-0.000860189, viewOffsetY=-7.70815e-06)
session.viewports['Viewport: 2'].view.setValues(width=0.0371357, 
    height=0.154989, viewOffsetX=6.72559e-05, viewOffsetY=-7.89106e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.0376547, 
    height=0.155482, viewOffsetX=-7.54962e-05, viewOffsetY=6.34746e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0368391, 
    height=0.155637, viewOffsetX=0.000138515, viewOffsetY=0.000134564)
session.viewports['Viewport: 10'].view.setValues(width=0.0400521, 
    height=0.155326, viewOffsetX=-0.000718314, viewOffsetY=-7.71752e-06)
session.viewports['Viewport: 11'].view.setValues(width=0.0365157, 
    height=0.15427, viewOffsetX=0.000137299, viewOffsetY=0.000133384)
session.viewports['Viewport: 12'].view.setValues(width=0.0385182, 
    height=0.154197, viewOffsetX=-0.000359205, viewOffsetY=-7.85082e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.0383168, 
    height=0.146667, viewOffsetX=-0.000960333, viewOffsetY=-3.15129e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0351049, 
    height=0.146513, viewOffsetX=7.55819e-05, viewOffsetY=-0.000111058)
session.viewports['Viewport: 3'].view.setValues(width=0.0355949, 
    height=0.146977, viewOffsetX=-8.38658e-05, viewOffsetY=4.79849e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0348236, 
    height=0.147121, viewOffsetX=0.000155171, viewOffsetY=0.000127403)
session.viewports['Viewport: 10'].view.setValues(width=0.0378614, 
    height=0.146831, viewOffsetX=-0.000801864, viewOffsetY=-3.15492e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.0345179, 
    height=0.14583, viewOffsetX=0.000153809, viewOffsetY=0.000126285)
session.viewports['Viewport: 12'].view.setValues(width=0.0364118, 
    height=0.145765, viewOffsetX=-0.000400761, viewOffsetY=-0.000110492)
session.viewports['Viewport: 1'].view.setValues(width=0.03621, height=0.138603, 
    viewOffsetX=-0.00105551, viewOffsetY=-5.41357e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.033175, 
    height=0.138459, viewOffsetX=8.34951e-05, viewOffsetY=-0.000141611)
session.viewports['Viewport: 3'].view.setValues(width=0.0336374, 
    height=0.138894, viewOffsetX=-9.18198e-05, viewOffsetY=3.32649e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0329083, 
    height=0.13903, viewOffsetX=0.000171, viewOffsetY=0.000120597)
session.viewports['Viewport: 10'].view.setValues(width=0.0357797, 
    height=0.138758, viewOffsetX=-0.000881269, viewOffsetY=-5.41972e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.0326195, 
    height=0.13781, viewOffsetX=0.000169499, viewOffsetY=0.00011954)
session.viewports['Viewport: 12'].view.setValues(width=0.03441, 
    height=0.137751, viewOffsetX=-0.000440257, viewOffsetY=-0.000140888)
session.viewports['Viewport: 1'].view.setValues(width=0.0342091, 
    height=0.130944, viewOffsetX=-0.00114591, viewOffsetY=-7.56224e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0313421, 
    height=0.130809, viewOffsetX=9.10112e-05, viewOffsetY=-0.00017063)
session.viewports['Viewport: 3'].view.setValues(width=0.0317784, 
    height=0.131218, viewOffsetX=-9.93744e-05, viewOffsetY=1.92842e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0310894, 
    height=0.131345, viewOffsetX=0.000186035, viewOffsetY=0.000114135)
session.viewports['Viewport: 10'].view.setValues(width=0.0338026, 
    height=0.13109, viewOffsetX=-0.000956689, viewOffsetY=-7.57083e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.0308165, 
    height=0.130193, viewOffsetX=0.000184402, viewOffsetY=0.000113134)
session.viewports['Viewport: 12'].view.setValues(width=0.0325088, 
    height=0.13014, viewOffsetX=-0.000477771, viewOffsetY=-0.000169759)
session.viewports['Viewport: 1'].view.setValues(width=0.0323099, 
    height=0.123674, viewOffsetX=-0.00123172, viewOffsetY=-9.6019e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0296022, 
    height=0.123547, viewOffsetX=9.81461e-05, viewOffsetY=-0.000198177)
session.viewports['Viewport: 3'].view.setValues(width=0.0300139, 
    height=0.123932, viewOffsetX=-0.000106546, viewOffsetY=6.01322e-06)
session.viewports['Viewport: 4'].view.setValues(width=0.0293629, 
    height=0.124051, viewOffsetX=0.000200306, viewOffsetY=0.000108001)
session.viewports['Viewport: 10'].view.setValues(width=0.0319259, 
    height=0.123812, viewOffsetX=-0.00102828, viewOffsetY=-9.61277e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.0291052, 
    height=0.122963, viewOffsetX=0.000198548, viewOffsetY=0.000107054)
session.viewports['Viewport: 12'].view.setValues(width=0.0307042, 
    height=0.122916, viewOffsetX=-0.000513383, viewOffsetY=-0.000197165)
session.viewports['Viewport: 1'].view.setValues(width=0.0305081, 
    height=0.116777, viewOffsetX=-0.00131314, viewOffsetY=-0.00011537)
session.viewports['Viewport: 2'].view.setValues(width=0.0279516, 
    height=0.116658, viewOffsetX=0.000104916, viewOffsetY=-0.000224313)
session.viewports['Viewport: 3'].view.setValues(width=0.0283399, 
    height=0.11702, viewOffsetX=-0.00011335, viewOffsetY=-6.57709e-06)
session.viewports['Viewport: 4'].view.setValues(width=0.027725, 
    height=0.117132, viewOffsetX=0.000213846, viewOffsetY=0.000102182)
session.viewports['Viewport: 10'].view.setValues(width=0.0301455, 
    height=0.116908, viewOffsetX=-0.00109621, viewOffsetY=-0.0001155)
session.viewports['Viewport: 11'].view.setValues(width=0.0274817, 
    height=0.116104, viewOffsetX=0.000211969, viewOffsetY=0.000101286)
session.viewports['Viewport: 12'].view.setValues(width=0.0289921, 
    height=0.116062, viewOffsetX=-0.000547171, viewOffsetY=-0.000223167)
session.viewports['Viewport: 1'].view.setValues(width=0.0287996, 
    height=0.110237, viewOffsetX=-0.00139034, viewOffsetY=-0.00013372)
session.viewports['Viewport: 2'].view.setValues(width=0.0263864, 
    height=0.110126, viewOffsetX=0.000111335, viewOffsetY=-0.000249097)
session.viewports['Viewport: 3'].view.setValues(width=0.0267526, 
    height=0.110466, viewOffsetX=-0.000119802, viewOffsetY=-1.85156e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.026172, 
    height=0.110571, viewOffsetX=0.000226685, viewOffsetY=9.66637e-05)
session.viewports['Viewport: 10'].view.setValues(width=0.0284573, 
    height=0.110361, viewOffsetX=-0.00116062, viewOffsetY=-0.000133871)
session.viewports['Viewport: 11'].view.setValues(width=0.0259423, 
    height=0.1096, viewOffsetX=0.000224695, viewOffsetY=9.58163e-05)
session.viewports['Viewport: 12'].view.setValues(width=0.0273687, 
    height=0.109563, viewOffsetX=-0.000579211, viewOffsetY=-0.000247824)
session.viewports['Viewport: 1'].view.setValues(width=0.0271803, 
    height=0.104039, viewOffsetX=-0.00146352, viewOffsetY=-0.000151112)
session.viewports['Viewport: 2'].view.setValues(width=0.024903, 
    height=0.103935, viewOffsetX=0.00011742, viewOffsetY=-0.000272587)
session.viewports['Viewport: 3'].view.setValues(width=0.0252483, 
    height=0.104254, viewOffsetX=-0.000125917, viewOffsetY=-2.98307e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0247002, 
    height=0.104353, viewOffsetX=0.000238854, viewOffsetY=9.14342e-05)
session.viewports['Viewport: 10'].view.setValues(width=0.0268573, 
    height=0.104156, viewOffsetX=-0.00122168, viewOffsetY=-0.000151282)
session.viewports['Viewport: 11'].view.setValues(width=0.0244834, 
    height=0.103437, viewOffsetX=0.000236757, viewOffsetY=9.06327e-05)
session.viewports['Viewport: 12'].view.setValues(width=0.0258301, 
    height=0.103404, viewOffsetX=-0.000609581, viewOffsetY=-0.000271195)
session.viewports['Viewport: 1'].view.setValues(width=0.0256463, 
    height=0.0981677, viewOffsetX=-0.00153285, viewOffsetY=-0.000167589)
session.viewports['Viewport: 2'].view.setValues(width=0.0234977, 
    height=0.0980696, viewOffsetX=0.000123184, viewOffsetY=-0.000294842)
session.viewports['Viewport: 3'].view.setValues(width=0.0238232, 
    height=0.0983699, viewOffsetX=-0.00013171, viewOffsetY=-4.055e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0233059, 
    height=0.0984621, viewOffsetX=0.000250382, viewOffsetY=8.64803e-05)
session.viewports['Viewport: 10'].view.setValues(width=0.0253416, 
    height=0.0982773, viewOffsetX=-0.00127952, viewOffsetY=-0.000167777)
session.viewports['Viewport: 11'].view.setValues(width=0.0231014, 
    height=0.0975979, viewOffsetX=0.000248185, viewOffsetY=8.57223e-05)
session.viewports['Viewport: 12'].view.setValues(width=0.0243724, 
    height=0.0975686, viewOffsetX=-0.000638352, viewOffsetY=-0.000293336)
session.viewports['Viewport: 1'].view.setValues(width=0.0241938, 
    height=0.0926078, viewOffsetX=-0.0015985, viewOffsetY=-0.000183191)
session.viewports['Viewport: 2'].view.setValues(width=0.022167, 
    height=0.0925158, viewOffsetX=0.000128643, viewOffsetY=-0.000315916)
session.viewports['Viewport: 3'].view.setValues(width=0.0224738, 
    height=0.092798, viewOffsetX=-0.000137196, viewOffsetY=-5.07003e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0219857, 
    height=0.0928844, viewOffsetX=0.000261299, viewOffsetY=8.17895e-05)
session.viewports['Viewport: 10'].view.setValues(width=0.0239063, 
    height=0.0927112, viewOffsetX=-0.00133429, viewOffsetY=-0.000183397)
session.viewports['Viewport: 11'].view.setValues(width=0.0217927, 
    height=0.0920692, viewOffsetX=0.000259006, viewOffsetY=8.10729e-05)
session.viewports['Viewport: 12'].view.setValues(width=0.0229922, 
    height=0.0920431, viewOffsetX=-0.000665599, viewOffsetY=-0.000314302)
session.viewports['Viewport: 1'].view.setValues(width=0.022819, 
    height=0.0873452, viewOffsetX=-0.00166064, viewOffsetY=-0.00019796)
session.viewports['Viewport: 2'].view.setValues(width=0.0209074, 
    height=0.0872589, viewOffsetX=0.00013381, viewOffsetY=-0.000335865)
session.viewports['Viewport: 3'].view.setValues(width=0.0211966, 
    height=0.0875241, viewOffsetX=-0.000142389, viewOffsetY=-6.03081e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0207361, 
    height=0.0876052, viewOffsetX=0.000271632, viewOffsetY=7.73498e-05)
session.viewports['Viewport: 10'].view.setValues(width=0.0225478, 
    height=0.0874427, viewOffsetX=-0.00138614, viewOffsetY=-0.000198183)
session.viewports['Viewport: 11'].view.setValues(width=0.0205541, 
    height=0.0868363, viewOffsetX=0.000269248, viewOffsetY=7.6672e-05)
session.viewports['Viewport: 12'].view.setValues(width=0.0216857, 
    height=0.086813, viewOffsetX=-0.00069139, viewOffsetY=-0.000334149)
session.viewports['Viewport: 1'].view.setValues(width=0.0215181, 
    height=0.0823659, viewOffsetX=-0.00171944, viewOffsetY=-0.000211935)
session.viewports['Viewport: 2'].view.setValues(width=0.0197157, 
    height=0.0822849, viewOffsetX=0.0001387, viewOffsetY=-0.000354741)
session.viewports['Viewport: 3'].view.setValues(width=0.0199881, 
    height=0.0825342, viewOffsetX=-0.000147303, viewOffsetY=-6.93987e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0195538, 
    height=0.0826102, viewOffsetX=0.00028141, viewOffsetY=7.31492e-05)
session.viewports['Viewport: 10'].view.setValues(width=0.0212624, 
    height=0.0824579, viewOffsetX=-0.00143519, viewOffsetY=-0.000212173)
session.viewports['Viewport: 11'].view.setValues(width=0.0193822, 
    height=0.0818852, viewOffsetX=0.00027894, viewOffsetY=7.2508e-05)
session.viewports['Viewport: 12'].view.setValues(width=0.0204496, 
    height=0.0818645, viewOffsetX=-0.000715794, viewOffsetY=-0.000352928)
session.viewports['Viewport: 1'].view.setValues(width=0.0202878, 
    height=0.0776565, viewOffsetX=-0.0016237, viewOffsetY=-0.000183216)
session.viewports['Viewport: 2'].view.setValues(width=0.0185885, 
    height=0.0775805, viewOffsetX=0.000294792, viewOffsetY=-0.000330629)
session.viewports['Viewport: 3'].view.setValues(width=0.0188452, 
    height=0.0778147, viewOffsetX=-5.3877e-07, viewOffsetY=-3.60465e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0184356, 
    height=0.0778861, viewOffsetX=0.000441952, viewOffsetY=0.000111094)
session.viewports['Viewport: 10'].view.setValues(width=0.0200467, 
    height=0.0777432, viewOffsetX=-0.00133007, viewOffsetY=-0.000183422)
session.viewports['Viewport: 11'].view.setValues(width=0.0182738, 
    height=0.0772026, viewOffsetX=0.000438074, viewOffsetY=0.00011012)
session.viewports['Viewport: 12'].view.setValues(width=0.0192805, 
    height=0.0771841, viewOffsetX=-0.000588183, viewOffsetY=-0.000328939)
session.viewports['Viewport: 1'].view.setValues(width=0.0191245, 
    height=0.0732038, viewOffsetX=-0.00153318, viewOffsetY=-0.000156064)
session.viewports['Viewport: 2'].view.setValues(width=0.0175227, 
    height=0.0731326, viewOffsetX=0.000442378, viewOffsetY=-0.000307831)
session.viewports['Viewport: 3'].view.setValues(width=0.0177645, 
    height=0.0733526, viewOffsetX=0.000138224, viewOffsetY=-4.51272e-06)
session.viewports['Viewport: 4'].view.setValues(width=0.0173784, 
    height=0.0734196, viewOffsetX=0.000593742, viewOffsetY=0.00014697)
session.viewports['Viewport: 10'].view.setValues(width=0.0188972, 
    height=0.0732856, viewOffsetX=-0.00123067, viewOffsetY=-0.000156239)
session.viewports['Viewport: 11'].view.setValues(width=0.0172259, 
    height=0.0727753, viewOffsetX=0.000588531, viewOffsetY=0.000145681)
session.viewports['Viewport: 12'].view.setValues(width=0.018175, 
    height=0.0727589, viewOffsetX=-0.000467527, viewOffsetY=-0.000306258)
session.viewports['Viewport: 1'].view.setValues(width=0.0180251, 
    height=0.0689953, viewOffsetX=-0.00147751, viewOffsetY=-0.000167658)
session.viewports['Viewport: 2'].view.setValues(width=0.0165154, 
    height=0.0689285, viewOffsetX=0.00055197, viewOffsetY=-0.000323568)
session.viewports['Viewport: 3'].view.setValues(width=0.0167432, 
    height=0.0691353, viewOffsetX=0.000239485, viewOffsetY=-1.19788e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0163791, 
    height=0.0691981, viewOffsetX=0.000707337, viewOffsetY=0.000143637)
session.viewports['Viewport: 10'].view.setValues(width=0.0178108, 
    height=0.0690724, viewOffsetX=-0.00116664, viewOffsetY=-0.000167847)
session.viewports['Viewport: 11'].view.setValues(width=0.0162354, 
    height=0.0685908, viewOffsetX=0.00070113, viewOffsetY=0.000142377)
session.viewports['Viewport: 12'].view.setValues(width=0.0171302, 
    height=0.0685763, viewOffsetX=-0.000383238, viewOffsetY=-0.000321915)
session.viewports['Viewport: 1'].view.setValues(width=0.0192239, 
    height=0.073584, viewOffsetX=-0.00154114, viewOffsetY=-0.000156874)
session.viewports['Viewport: 2'].view.setValues(width=0.0176139, 
    height=0.073513, viewOffsetX=0.00044468, viewOffsetY=-0.000309432)
session.viewports['Viewport: 3'].view.setValues(width=0.0178566, 
    height=0.073733, viewOffsetX=0.000138941, viewOffsetY=-4.53609e-06)
session.viewports['Viewport: 4'].view.setValues(width=0.0174683, 
    height=0.0737997, viewOffsetX=0.000596815, viewOffsetY=0.000147731)
session.viewports['Viewport: 10'].view.setValues(width=0.0189954, 
    height=0.0736663, viewOffsetX=-0.00123706, viewOffsetY=-0.000157051)
session.viewports['Viewport: 11'].view.setValues(width=0.017315, 
    height=0.073152, viewOffsetX=0.000591579, viewOffsetY=0.000146435)
session.viewports['Viewport: 12'].view.setValues(width=0.0182696, 
    height=0.0731374, viewOffsetX=-0.000469959, viewOffsetY=-0.000307851)
session.viewports['Viewport: 12'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM2', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 12'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM3', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 12'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM4', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 12'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM8', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 12'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM9', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 12'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM10', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 12'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM9', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 12'].view.setValues(session.views['Back'])
session.viewports['Viewport: 1'].view.setValues(width=0.0637301, 
    height=0.243943, viewOffsetY=-1.31732e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0583029, 
    height=0.243332, viewOffsetX=0.000171557, viewOffsetY=-2.6325e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.0590752, 
    height=0.243931, viewOffsetX=0.000145029)
session.viewports['Viewport: 4'].view.setValues(width=0.0577744, 
    height=0.244084, viewOffsetX=0.000184386, viewOffsetY=1.31357e-05)
session.viewports['Viewport: 10'].view.setValues(width=0.062961, 
    height=0.24417, viewOffsetX=2.64394e-05, viewOffsetY=-1.31866e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.0576129, 
    height=0.243401, viewOffsetX=0.000183871, viewOffsetY=1.3099e-05)
session.viewports['Viewport: 12'].view.setValues(width=0.0609049, 
    height=0.243817, viewOffsetX=9.25605e-05, viewOffsetY=-2.63777e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.0524406, 
    height=0.200729, viewOffsetY=-2.23717e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0480513, 
    height=0.200546, viewOffsetX=0.000291808, viewOffsetY=-4.47779e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.0487376, 
    height=0.201246, viewOffsetX=0.000246939)
session.viewports['Viewport: 4'].view.setValues(width=0.0476888, 
    height=0.201474, viewOffsetX=0.000314111, viewOffsetY=2.23779e-05)
session.viewports['Viewport: 10'].view.setValues(width=0.0518189, 
    height=0.200959, viewOffsetX=4.49098e-05, viewOffsetY=-2.2398e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.0475288, 
    height=0.200798, viewOffsetX=0.000313058, viewOffsetY=2.23026e-05)
session.viewports['Viewport: 12'].view.setValues(width=0.0501741, 
    height=0.200859, viewOffsetX=0.000157372, viewOffsetY=-4.48477e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.0507131, 
    height=0.194117, viewOffsetY=-3.34991e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0464589, height=0.1939, 
    viewOffsetX=0.000436852, viewOffsetY=-6.70345e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.0471103, 
    height=0.194526, viewOffsetX=0.000369584)
session.viewports['Viewport: 4'].view.setValues(width=0.0460908, 
    height=0.194723, viewOffsetX=0.000470061, viewOffsetY=3.3488e-05)
session.viewports['Viewport: 10'].view.setValues(width=0.0501102, 
    height=0.194333, viewOffsetX=6.72436e-05, viewOffsetY=-3.3537e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.0459402, 
    height=0.194087, viewOffsetX=0.000468527, viewOffsetY=3.33782e-05)
session.viewports['Viewport: 12'].view.setValues(width=0.0485145, 
    height=0.194215, viewOffsetX=0.000235609, viewOffsetY=-6.71437e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.0478744, 
    height=0.183251, viewOffsetY=-4.35389e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0438598, 
    height=0.183052, viewOffsetX=0.000567795, viewOffsetY=-8.71278e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.0444756, 
    height=0.183647, viewOffsetX=0.000480373)
session.viewports['Viewport: 4'].view.setValues(width=0.0435135, 
    height=0.183834, viewOffsetX=0.000610976, viewOffsetY=4.35267e-05)
session.viewports['Viewport: 10'].view.setValues(width=0.0473054, 
    height=0.183456, viewOffsetX=8.73968e-05, viewOffsetY=-4.35879e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.0433709, 
    height=0.183232, viewOffsetX=0.000608973, viewOffsetY=4.33839e-05)
session.viewports['Viewport: 12'].view.setValues(width=0.0458, height=0.183348, 
    viewOffsetX=0.000306229, viewOffsetY=-8.72687e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.0453172, 
    height=0.173463, viewOffsetY=-5.32116e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0415173, 
    height=0.173276, viewOffsetX=0.000693941, viewOffsetY=-0.000106485)
session.viewports['Viewport: 3'].view.setValues(width=0.0420991, 
    height=0.173834, viewOffsetX=0.000587082)
session.viewports['Viewport: 4'].view.setValues(width=0.0411879, 
    height=0.174009, viewOffsetX=0.000746687, viewOffsetY=5.31952e-05)
session.viewports['Viewport: 10'].view.setValues(width=0.0447786, 
    height=0.173656, viewOffsetX=0.000106813, viewOffsetY=-5.32711e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.041053, 
    height=0.173439, viewOffsetX=0.000744242, viewOffsetY=5.30206e-05)
session.viewports['Viewport: 12'].view.setValues(width=0.043354, 
    height=0.173556, viewOffsetX=0.000374264, viewOffsetY=-0.000106657)
session.viewports['Viewport: 1'].view.setValues(width=0.0428651, 
    height=0.164077, viewOffsetY=-6.24057e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0392712, 
    height=0.163902, viewOffsetX=0.000813853, viewOffsetY=-0.000124885)
session.viewports['Viewport: 3'].view.setValues(width=0.0398208, 
    height=0.164427, viewOffsetX=0.000688515)
session.viewports['Viewport: 4'].view.setValues(width=0.0389586, 
    height=0.164591, viewOffsetX=0.000875689, viewOffsetY=6.23851e-05)
session.viewports['Viewport: 10'].view.setValues(width=0.0423556, 
    height=0.16426, viewOffsetX=0.000125269, viewOffsetY=-6.24754e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.0388309, 
    height=0.164052, viewOffsetX=0.000872821, viewOffsetY=6.21808e-05)
session.viewports['Viewport: 12'].view.setValues(width=0.0410085, 
    height=0.164167, viewOffsetX=0.000438936, viewOffsetY=-0.000125087)
session.viewports['Viewport: 1'].view.setValues(width=0.0405338, 
    height=0.155153, viewOffsetY=-7.11571e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0371357, 
    height=0.154989, viewOffsetX=0.000927992, viewOffsetY=-0.0001424)
session.viewports['Viewport: 3'].view.setValues(width=0.0376547, 
    height=0.155482, viewOffsetX=0.000785062)
session.viewports['Viewport: 4'].view.setValues(width=0.0368391, 
    height=0.155637, viewOffsetX=0.000998474, viewOffsetY=7.11322e-05)
session.viewports['Viewport: 10'].view.setValues(width=0.0400521, 
    height=0.155326, viewOffsetX=0.000142836, viewOffsetY=-7.12367e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.0367184, 
    height=0.155127, viewOffsetX=0.000995203, viewOffsetY=7.08999e-05)
session.viewports['Viewport: 12'].view.setValues(width=0.0387785, 
    height=0.155239, viewOffsetX=0.000500495, viewOffsetY=-0.00014263)
session.viewports['Viewport: 1'].view.setValues(width=0.0383168, 
    height=0.146667, viewOffsetY=-7.94793e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0351049, 
    height=0.146513, viewOffsetX=0.00103653, viewOffsetY=-0.000159055)
session.viewports['Viewport: 3'].view.setValues(width=0.0355949, 
    height=0.146977, viewOffsetX=0.00087687)
session.viewports['Viewport: 4'].view.setValues(width=0.0348236, 
    height=0.147121, viewOffsetX=0.00111523, viewOffsetY=7.94498e-05)
session.viewports['Viewport: 10'].view.setValues(width=0.0378614, 
    height=0.146831, viewOffsetX=0.000159541, viewOffsetY=-7.95678e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.0347095, 
    height=0.14664, viewOffsetX=0.00111158, viewOffsetY=7.91907e-05)
session.viewports['Viewport: 12'].view.setValues(width=0.0366578, 
    height=0.14675, viewOffsetX=0.000559034, viewOffsetY=-0.000159313)
session.viewports['Viewport: 1'].view.setValues(width=0.03621, height=0.138603, 
    viewOffsetY=-8.73886e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.033175, 
    height=0.138459, viewOffsetX=0.00113969, viewOffsetY=-0.000174885)
session.viewports['Viewport: 3'].view.setValues(width=0.0336374, 
    height=0.138894, viewOffsetX=0.000964122)
session.viewports['Viewport: 4'].view.setValues(width=0.0329083, 
    height=0.13903, viewOffsetX=0.00122619, viewOffsetY=8.7355e-05)
session.viewports['Viewport: 10'].view.setValues(width=0.0357797, 
    height=0.138758, viewOffsetX=0.000175417, viewOffsetY=-8.74856e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.0328005, 
    height=0.138575, viewOffsetX=0.00122217, viewOffsetY=8.70699e-05)
session.viewports['Viewport: 12'].view.setValues(width=0.0346425, 
    height=0.138682, viewOffsetX=0.00061467, viewOffsetY=-0.000175168)
session.viewports['Viewport: 1'].view.setValues(width=0.0342091, 
    height=0.130944, viewOffsetY=-9.49007e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0313421, 
    height=0.130809, viewOffsetX=0.00123768, viewOffsetY=-0.00018992)
session.viewports['Viewport: 3'].view.setValues(width=0.0317784, 
    height=0.131218, viewOffsetX=0.00104699)
session.viewports['Viewport: 4'].view.setValues(width=0.0310894, 
    height=0.131345, viewOffsetX=0.00133158, viewOffsetY=9.48633e-05)
session.viewports['Viewport: 10'].view.setValues(width=0.0338026, 
    height=0.13109, viewOffsetX=0.000190497, viewOffsetY=-9.50062e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.0309876, 
    height=0.130915, viewOffsetX=0.00132722, viewOffsetY=9.45535e-05)
session.viewports['Viewport: 12'].view.setValues(width=0.0327285, 
    height=0.13102, viewOffsetX=0.000667515, viewOffsetY=-0.000190228)
session.viewports['Viewport: 1'].view.setValues(width=0.0323099, 
    height=0.123674, viewOffsetY=-0.000102032)
session.viewports['Viewport: 2'].view.setValues(width=0.0296022, 
    height=0.123547, viewOffsetX=0.00133069, viewOffsetY=-0.000204193)
session.viewports['Viewport: 3'].view.setValues(width=0.0300139, 
    height=0.123932, viewOffsetX=0.00112566)
session.viewports['Viewport: 4'].view.setValues(width=0.0293629, 
    height=0.124051, viewOffsetX=0.00143162, viewOffsetY=0.00010199)
session.viewports['Viewport: 10'].view.setValues(width=0.0319259, 
    height=0.123812, viewOffsetX=0.000204812, viewOffsetY=-0.000102145)
session.viewports['Viewport: 11'].view.setValues(width=0.0292667, 
    height=0.123645, viewOffsetX=0.00142693, viewOffsetY=0.000101657)
session.viewports['Viewport: 12'].view.setValues(width=0.0309117, 
    height=0.123747, viewOffsetX=0.00071768, viewOffsetY=-0.000204524)
session.viewports['Viewport: 1'].view.setValues(width=0.0305081, 
    height=0.116777, viewOffsetX=-1.2645e-05, viewOffsetY=-9.61854e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0279516, 
    height=0.116658, viewOffsetX=0.00140629)
session.viewports['Viewport: 3'].view.setValues(width=0.0283399, 
    height=0.11702, viewOffsetX=0.00118765, viewOffsetY=1.26171e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.027725, 
    height=0.117132, viewOffsetX=0.00151389, viewOffsetY=0.00012136)
session.viewports['Viewport: 10'].view.setValues(width=0.0301455, 
    height=0.116908, viewOffsetX=0.000205734, viewOffsetY=-9.62924e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.0276342, 
    height=0.116748, viewOffsetX=0.00150893, viewOffsetY=0.000120963)
session.viewports['Viewport: 12'].view.setValues(width=0.029188, 
    height=0.116846, viewOffsetX=0.000752602)
session.viewports['Viewport: 1'].view.setValues(width=0.0287996, 
    height=0.110237, viewOffsetX=-2.46357e-05, viewOffsetY=-9.06416e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0263864, 
    height=0.110126, viewOffsetX=0.00147797)
session.viewports['Viewport: 3'].view.setValues(width=0.0267526, 
    height=0.110466, viewOffsetX=0.00124642, viewOffsetY=2.45813e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.026172, 
    height=0.110571, viewOffsetX=0.0015919, viewOffsetY=0.000139727)
session.viewports['Viewport: 10'].view.setValues(width=0.0284573, 
    height=0.110361, viewOffsetX=0.000206609, viewOffsetY=-9.07424e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.0260863, 
    height=0.110209, viewOffsetX=0.00158669, viewOffsetY=0.00013927)
session.viewports['Viewport: 12'].view.setValues(width=0.0275536, 
    height=0.110304, viewOffsetX=0.000785718)
session.viewports['Viewport: 1'].view.setValues(width=0.0271803, 
    height=0.104039, viewOffsetX=-3.60003e-05, viewOffsetY=-8.53875e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.024903, 
    height=0.103935, viewOffsetX=0.00154592)
session.viewports['Viewport: 3'].view.setValues(width=0.0252483, 
    height=0.104254, viewOffsetX=0.00130214, viewOffsetY=3.59205e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0247002, 
    height=0.104353, viewOffsetX=0.00166584, viewOffsetY=0.000157135)
session.viewports['Viewport: 10'].view.setValues(width=0.0268573, 
    height=0.104156, viewOffsetX=0.000207439, viewOffsetY=-8.54822e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.0246193, 
    height=0.104011, viewOffsetX=0.00166039, viewOffsetY=0.000156621)
session.viewports['Viewport: 12'].view.setValues(width=0.0260046, 
    height=0.104102, viewOffsetX=0.000817107)
session.viewports['Viewport: 1'].view.setValues(width=0.0256463, 
    height=0.0981677, viewOffsetX=-4.67667e-05, viewOffsetY=-8.041e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0234977, 
    height=0.0980696, viewOffsetX=0.0016103)
session.viewports['Viewport: 3'].view.setValues(width=0.0238232, 
    height=0.0983699, viewOffsetX=0.00135492, viewOffsetY=4.66628e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0233059, 
    height=0.0984621, viewOffsetX=0.00173589, viewOffsetY=0.000173626)
session.viewports['Viewport: 10'].view.setValues(width=0.0253416, 
    height=0.0982773, viewOffsetX=0.000208225, viewOffsetY=-8.04993e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.0232296, 
    height=0.0981397, viewOffsetX=0.00173021, viewOffsetY=0.000173058)
session.viewports['Viewport: 12'].view.setValues(width=0.0245371, 
    height=0.0982277, viewOffsetX=0.000846845)
session.viewports['Viewport: 1'].view.setValues(width=0.0241938, 
    height=0.0926078, viewOffsetX=-5.69619e-05, viewOffsetY=-7.5697e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.022167, 
    height=0.0925158, viewOffsetX=0.00167126)
session.viewports['Viewport: 3'].view.setValues(width=0.0224738, 
    height=0.092798, viewOffsetX=0.0014049, viewOffsetY=5.6835e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0219857, 
    height=0.0928844, viewOffsetX=0.00180223, viewOffsetY=0.000189242)
session.viewports['Viewport: 10'].view.setValues(width=0.0239063, 
    height=0.0927112, viewOffsetX=0.00020897, viewOffsetY=-7.57809e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.0219137, 
    height=0.0925803, viewOffsetX=0.00179633, viewOffsetY=0.000188623)
session.viewports['Viewport: 12'].view.setValues(width=0.0231475, 
    height=0.0926649, viewOffsetX=0.000875007)
session.viewports['Viewport: 1'].view.setValues(width=0.022819, 
    height=0.0873452, viewOffsetX=-6.66121e-05, viewOffsetY=-7.12358e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0209074, 
    height=0.0872589, viewOffsetX=0.00172897)
session.viewports['Viewport: 3'].view.setValues(width=0.0211966, 
    height=0.0875241, viewOffsetX=0.00145221, viewOffsetY=6.64636e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0207361, 
    height=0.0876052, viewOffsetX=0.00186502, viewOffsetY=0.000204024)
session.viewports['Viewport: 10'].view.setValues(width=0.0225478, 
    height=0.0874427, viewOffsetX=0.000209675, viewOffsetY=-7.13147e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.0206682, 
    height=0.0873184, viewOffsetX=0.00185891, viewOffsetY=0.000203356)
session.viewports['Viewport: 12'].view.setValues(width=0.0218322, 
    height=0.0873995, viewOffsetX=0.000901665)
session.viewports['Viewport: 1'].view.setValues(width=0.0215181, 
    height=0.0823659, viewOffsetX=-7.57431e-05, viewOffsetY=-6.70151e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0197157, 
    height=0.0822849, viewOffsetX=0.00178357)
session.viewports['Viewport: 3'].view.setValues(width=0.0199881, 
    height=0.0825342, viewOffsetX=0.00149697, viewOffsetY=7.55738e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0195538, 
    height=0.0826102, viewOffsetX=0.00192443, viewOffsetY=0.000218009)
session.viewports['Viewport: 10'].view.setValues(width=0.0212624, 
    height=0.0824579, viewOffsetX=0.000210343, viewOffsetY=-6.70889e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.0194898, 
    height=0.0823398, viewOffsetX=0.00191813, viewOffsetY=0.000217296)
session.viewports['Viewport: 12'].view.setValues(width=0.0205877, 
    height=0.0824175, viewOffsetX=0.000926889)
session.viewports['Viewport: 1'].view.setValues(width=0.0202878, 
    height=0.0776565, viewOffsetX=-8.43795e-05, viewOffsetY=-6.3023e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0185885, 
    height=0.0775805, viewOffsetX=0.00183522)
session.viewports['Viewport: 3'].view.setValues(width=0.0188452, 
    height=0.0778147, viewOffsetX=0.00153931, viewOffsetY=8.41906e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0184356, 
    height=0.0778861, viewOffsetX=0.00198062, viewOffsetY=0.000231237)
session.viewports['Viewport: 10'].view.setValues(width=0.0200467, 
    height=0.0777432, viewOffsetX=0.000210975, viewOffsetY=-6.30923e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.0183752, 
    height=0.0776311, viewOffsetX=0.00197414, viewOffsetY=0.000230481)
session.viewports['Viewport: 12'].view.setValues(width=0.0194107, 
    height=0.0777055, viewOffsetX=0.000950748)
session.viewports['Viewport: 1'].view.setValues(width=0.0191245, 
    height=0.0732038, viewOffsetX=-9.25452e-05, viewOffsetY=-5.92483e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0175227, 
    height=0.0731326, viewOffsetX=0.00188405)
session.viewports['Viewport: 3'].view.setValues(width=0.0177645, 
    height=0.0733526, viewOffsetX=0.00157935, viewOffsetY=9.23376e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0173784, 
    height=0.0734196, viewOffsetX=0.00203375, viewOffsetY=0.000243744)
session.viewports['Viewport: 10'].view.setValues(width=0.0188972, 
    height=0.0732856, viewOffsetX=0.000211572, viewOffsetY=-5.93136e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.0173215, 
    height=0.0731793, viewOffsetX=0.00202709, viewOffsetY=0.000242947)
session.viewports['Viewport: 12'].view.setValues(width=0.0182978, 
    height=0.0732504, viewOffsetX=0.000973308)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.482485, 
    farPlane=0.525352, width=0.0191734, height=0.0733907, 
    viewOffsetX=-0.0012995, viewOffsetY=0.000188542)
session.viewports['Viewport: 2'].view.setValues(nearPlane=0.482015, 
    farPlane=0.524912, width=0.0175675, height=0.0733196, 
    viewOffsetX=0.000783218, viewOffsetY=3.62188e-05)
session.viewports['Viewport: 3'].view.setValues(nearPlane=0.483524, 
    farPlane=0.526404, width=0.0178098, height=0.0735396, 
    viewOffsetX=0.000462476, viewOffsetY=0.000341021)
session.viewports['Viewport: 4'].view.setValues(nearPlane=0.483923, 
    farPlane=0.526815, width=0.0174226, height=0.0736065, 
    viewOffsetX=0.000942397, viewOffsetY=0.000493036)
session.viewports['Viewport: 10'].view.setValues(nearPlane=0.483019, 
    farPlane=0.525934, width=0.0189455, height=0.0734727, 
    viewOffsetX=-0.000980262, viewOffsetY=0.000188756)
session.viewports['Viewport: 11'].view.setValues(nearPlane=0.482317, 
    farPlane=0.525025, width=0.0173656, height=0.0733655, 
    viewOffsetX=0.000939314, viewOffsetY=0.000491425)
session.viewports['Viewport: 12'].view.setValues(nearPlane=0.48283, 
    farPlane=0.525795, width=0.0183446, height=0.0734377, 
    viewOffsetX=-0.000178758, viewOffsetY=3.62794e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.0180231, 
    height=0.0689877, viewOffsetX=-0.00123974, viewOffsetY=9.54067e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0165136, 
    height=0.0689208, viewOffsetX=0.000891619, viewOffsetY=-6.07127e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.0167413, 
    height=0.0691275, viewOffsetX=0.000563359, viewOffsetY=0.000251582)
session.viewports['Viewport: 4'].view.setValues(width=0.0163773, 
    height=0.0691905, viewOffsetX=0.00105441, viewOffsetY=0.000407392)
session.viewports['Viewport: 10'].view.setValues(width=0.0178089, 
    height=0.0690648, viewOffsetX=-0.00091296, viewOffsetY=9.55166e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.0163237, 
    height=0.0689641, viewOffsetX=0.00105096, viewOffsetY=0.00040606)
session.viewports['Viewport: 12'].view.setValues(width=0.017244, 
    height=0.0690319, viewOffsetX=-9.26512e-05, viewOffsetY=-6.08083e-05)
session.viewports['Viewport: 12'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM2', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 12'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM1', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 12'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM2', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 12'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM8', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 12'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM9', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].view.setValues(width=0.0169863, 
    height=0.0650193, viewOffsetX=-0.00117964, viewOffsetY=7.108e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0155637, 
    height=0.0649565, viewOffsetX=0.00100318, viewOffsetY=-8.89822e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.0157782, 
    height=0.0651507, viewOffsetX=0.000666961, viewOffsetY=0.00023117)
session.viewports['Viewport: 4'].view.setValues(width=0.0154351, 
    height=0.0652098, viewOffsetX=0.00116977, viewOffsetY=0.000390913)
session.viewports['Viewport: 10'].view.setValues(width=0.0167844, 
    height=0.0650918, viewOffsetX=-0.000844882, viewOffsetY=7.11626e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.0153846, 
    height=0.0649964, viewOffsetX=0.00116594, viewOffsetY=0.000389636)
session.viewports['Viewport: 12'].view.setValues(width=0.0162521, 
    height=0.0650611, viewOffsetX=-4.68385e-06, viewOffsetY=-8.91234e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.0160049, 
    height=0.0612628, viewOffsetX=-0.00112273, viewOffsetY=4.80902e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0146646, 
    height=0.0612039, viewOffsetX=0.00110845, viewOffsetY=-0.000115679)
session.viewports['Viewport: 3'].view.setValues(width=0.0148665, 
    height=0.0613864, viewOffsetX=0.000764757, viewOffsetY=0.00021186)
session.viewports['Viewport: 4'].view.setValues(width=0.0145432, 
    height=0.0614418, viewOffsetX=0.00127862, viewOffsetY=0.000375302)
session.viewports['Viewport: 10'].view.setValues(width=0.0158147, 
    height=0.0613312, viewOffsetX=-0.00078047, viewOffsetY=4.81472e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.0144957, 
    height=0.0612408, viewOffsetX=0.00127444, viewOffsetY=0.000374075)
session.viewports['Viewport: 12'].view.setValues(width=0.0153132, 
    height=0.0613025, viewOffsetX=7.84209e-05, viewOffsetY=-0.000115863)
session.viewports['Viewport: 1'].view.setValues(width=0.0150783, 
    height=0.0577158, viewOffsetX=-0.001069, viewOffsetY=2.63806e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0138156, 
    height=0.0576605, viewOffsetX=0.00120788, viewOffsetY=-0.00014089)
session.viewports['Viewport: 3'].view.setValues(width=0.0140057, 
    height=0.057832, viewOffsetX=0.000857113, viewOffsetY=0.000193627)
session.viewports['Viewport: 4'].view.setValues(width=0.0137011, 
    height=0.057884, viewOffsetX=0.00138142, viewOffsetY=0.000360562)
session.viewports['Viewport: 10'].view.setValues(width=0.0148991, 
    height=0.0577802, viewOffsetX=-0.000719649, viewOffsetY=2.64133e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.0136563, 
    height=0.0576946, viewOffsetX=0.0013769, viewOffsetY=0.000359384)
session.viewports['Viewport: 12'].view.setValues(width=0.0144267, 
    height=0.0577534, viewOffsetX=0.0001569, viewOffsetY=-0.000141115)
session.viewports['Viewport: 1'].view.setValues(width=0.0142034, 
    height=0.0543671, viewOffsetX=-0.00101827, viewOffsetY=5.88479e-06)
session.viewports['Viewport: 2'].view.setValues(width=0.013014, 
    height=0.0543152, viewOffsetX=0.00130174, viewOffsetY=-0.000164692)
session.viewports['Viewport: 3'].view.setValues(width=0.0131931, 
    height=0.0544764, viewOffsetX=0.000944304, viewOffsetY=0.000176413)
session.viewports['Viewport: 4'].view.setValues(width=0.0129061, 
    height=0.0545252, viewOffsetX=0.00147847, viewOffsetY=0.000346646)
session.viewports['Viewport: 10'].view.setValues(width=0.0140346, 
    height=0.0544278, viewOffsetX=-0.000662229, viewOffsetY=5.8946e-06)
session.viewports['Viewport: 11'].view.setValues(width=0.0128639, 
    height=0.0543468, viewOffsetX=0.00147363, viewOffsetY=0.000345514)
session.viewports['Viewport: 12'].view.setValues(width=0.0135897, 
    height=0.0544027, viewOffsetX=0.000230992, viewOffsetY=-0.000164955)
session.viewports['Viewport: 1'].view.setValues(width=0.0133777, 
    height=0.0512065, viewOffsetX=-0.000970384, viewOffsetY=-1.34601e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0122575, 
    height=0.0511578, viewOffsetX=0.00139034, viewOffsetY=-0.000187157)
session.viewports['Viewport: 3'].view.setValues(width=0.0124261, 
    height=0.0513092, viewOffsetX=0.0010266, viewOffsetY=0.000160166)
session.viewports['Viewport: 4'].view.setValues(width=0.0121557, 
    height=0.0513551, viewOffsetX=0.00157006, viewOffsetY=0.000333512)
session.viewports['Viewport: 10'].view.setValues(width=0.0132187, 
    height=0.0512637, viewOffsetX=-0.000608034, viewOffsetY=-1.34719e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.0121159, 
    height=0.051187, viewOffsetX=0.00156493, viewOffsetY=0.000332423)
session.viewports['Viewport: 12'].view.setValues(width=0.0127997, 
    height=0.0512402, viewOffsetX=0.000300924, viewOffsetY=-0.000187456)
session.viewports['Viewport: 1'].view.setValues(width=0.0125986, 
    height=0.0482241, viewOffsetX=-0.000925201, viewOffsetY=-3.17145e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0115436, 
    height=0.0481783, viewOffsetX=0.00147394, viewOffsetY=-0.000208357)
session.viewports['Viewport: 3'].view.setValues(width=0.0117023, 
    height=0.0483207, viewOffsetX=0.00110426, viewOffsetY=0.000144835)
session.viewports['Viewport: 4'].view.setValues(width=0.0114477, 
    height=0.0483637, viewOffsetX=0.0016565, viewOffsetY=0.000321119)
session.viewports['Viewport: 10'].view.setValues(width=0.0124488, 
    height=0.0482779, viewOffsetX=-0.000556894, viewOffsetY=-3.17467e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.0114102, 
    height=0.0482055, viewOffsetX=0.00165108, viewOffsetY=0.00032007)
session.viewports['Viewport: 12'].view.setValues(width=0.0120542, 
    height=0.0482559, viewOffsetX=0.000366914, viewOffsetY=-0.00020869)
session.viewports['Viewport: 1'].view.setValues(width=0.0118635, 
    height=0.0454105, viewOffsetX=-0.000916996, viewOffsetY=-3.91272e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0108702, 
    height=0.0453675, viewOffsetX=0.00151837, viewOffsetY=-0.00021854)
session.viewports['Viewport: 3'].view.setValues(width=0.0110195, 
    height=0.0455013, viewOffsetX=0.00114309, viewOffsetY=0.000140183)
session.viewports['Viewport: 4'].view.setValues(width=0.0107797, 
    height=0.0455417, viewOffsetX=0.00170364, viewOffsetY=0.000319231)
session.viewports['Viewport: 10'].view.setValues(width=0.0117225, 
    height=0.0454611, viewOffsetX=-0.000543106, viewOffsetY=-3.91676e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.0107444, 
    height=0.0453927, viewOffsetX=0.00169806, viewOffsetY=0.000318189)
session.viewports['Viewport: 12'].view.setValues(width=0.011351, 
    height=0.0454406, viewOffsetX=0.000394669, viewOffsetY=-0.00021889)
session.viewports['Viewport: 1'].view.setValues(width=0.0111702, 
    height=0.0427566, viewOffsetX=-0.000909257, viewOffsetY=-4.6119e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.0102349, 
    height=0.0427163, viewOffsetX=0.00156027, viewOffsetY=-0.000228146)
session.viewports['Viewport: 3'].view.setValues(width=0.0103755, 
    height=0.042842, viewOffsetX=0.00117971, viewOffsetY=0.000135796)
session.viewports['Viewport: 4'].view.setValues(width=0.0101496, 
    height=0.0428799, viewOffsetX=0.0017481, viewOffsetY=0.000317451)
session.viewports['Viewport: 10'].view.setValues(width=0.0110374, 
    height=0.0428043, viewOffsetX=-0.000530102, viewOffsetY=-4.61672e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.0101164, 
    height=0.0427396, viewOffsetX=0.00174238, viewOffsetY=0.000316414)
session.viewports['Viewport: 12'].view.setValues(width=0.0106876, 
    height=0.0427851, viewOffsetX=0.000420849, viewOffsetY=-0.000228511)
session.viewports['Viewport: 1'].view.setValues(width=0.0105164, 
    height=0.040254, viewOffsetX=-0.000901958, viewOffsetY=-5.27124e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.00963586, 
    height=0.0402161, viewOffsetX=0.00159979, viewOffsetY=-0.000237205)
session.viewports['Viewport: 3'].view.setValues(width=0.00976814, 
    height=0.0403342, viewOffsetX=0.00121425, viewOffsetY=0.000131659)
session.viewports['Viewport: 4'].view.setValues(width=0.00955551, 
    height=0.0403698, viewOffsetX=0.00179003, viewOffsetY=0.000315772)
session.viewports['Viewport: 10'].view.setValues(width=0.0103914, 
    height=0.0402989, viewOffsetX=-0.000517838, viewOffsetY=-5.2768e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.00952425, 
    height=0.0402377, viewOffsetX=0.00178417, viewOffsetY=0.000314741)
session.viewports['Viewport: 12'].view.setValues(width=0.0100621, 
    height=0.0402808, viewOffsetX=0.000445537, viewOffsetY=-0.000237584)
session.viewports['Viewport: 1'].view.setValues(width=0.00989991, 
    height=0.0378943, viewOffsetX=-0.000895078, viewOffsetY=-5.89291e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.00907105, 
    height=0.0378588, viewOffsetX=0.00163705, viewOffsetY=-0.000245745)
session.viewports['Viewport: 3'].view.setValues(width=0.00919553, 
    height=0.0379698, viewOffsetX=0.00124682, viewOffsetY=0.000127758)
session.viewports['Viewport: 4'].view.setValues(width=0.00899534, 
    height=0.0380033, viewOffsetX=0.00182956, viewOffsetY=0.000314189)
session.viewports['Viewport: 10'].view.setValues(width=0.00978225, 
    height=0.0379366, viewOffsetX=-0.000506276, viewOffsetY=-5.89916e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.00896591, 
    height=0.0378789, viewOffsetX=0.00182358, viewOffsetY=0.000313164)
session.viewports['Viewport: 12'].view.setValues(width=0.00947228, 
    height=0.0379197, viewOffsetX=0.000468815, viewOffsetY=-0.000246139)
session.viewports['Viewport: 1'].view.setValues(width=0.0093188, 
    height=0.03567, viewOffsetX=-0.000888592, viewOffsetY=-6.47894e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.00853861, 
    height=0.0356366, viewOffsetX=0.00167217, viewOffsetY=-0.000253797)
session.viewports['Viewport: 3'].view.setValues(width=0.00865573, 
    height=0.0357409, viewOffsetX=0.00127752, viewOffsetY=0.000124081)
session.viewports['Viewport: 4'].view.setValues(width=0.00846729, 
    height=0.0357723, viewOffsetX=0.00186683, viewOffsetY=0.000312697)
session.viewports['Viewport: 10'].view.setValues(width=0.00920804, 
    height=0.0357098, viewOffsetX=-0.000495376, viewOffsetY=-6.48585e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.00843958, 
    height=0.0356553, viewOffsetX=0.00186072, viewOffsetY=0.000311676)
session.viewports['Viewport: 12'].view.setValues(width=0.00891628, 
    height=0.035694, viewOffsetX=0.000490758, viewOffsetY=-0.000254203)
session.viewports['Viewport: 1'].view.setValues(width=0.00877108, 
    height=0.0335735, viewOffsetX=-0.000784322, viewOffsetY=-8.1191e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.00803677, 
    height=0.0335421, viewOffsetX=0.00180351, viewOffsetY=-0.000272272)
session.viewports['Viewport: 3'].view.setValues(width=0.00814697, 
    height=0.0336402, viewOffsetX=0.00140464, viewOffsetY=0.000109734)
session.viewports['Viewport: 4'].view.setValues(width=0.00796958, 
    height=0.0336697, viewOffsetX=0.00200006, viewOffsetY=0.000300419)
session.viewports['Viewport: 10'].view.setValues(width=0.00866683, 
    height=0.0336109, viewOffsetX=-0.000386837, viewOffsetY=-8.12784e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.00794351, 
    height=0.0335595, viewOffsetX=0.00199352, viewOffsetY=0.000299438)
session.viewports['Viewport: 12'].view.setValues(width=0.00839224, 
    height=0.0335961, viewOffsetX=0.00060983, viewOffsetY=-0.000272708)
session.viewports['Viewport: 1'].view.setValues(width=0.00825493, 
    height=0.0315978, viewOffsetX=-0.00068606, viewOffsetY=-9.66476e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.00756384, 
    height=0.0315683, viewOffsetX=0.00192728, viewOffsetY=-0.000289683)
session.viewports['Viewport: 3'].view.setValues(width=0.00766753, 
    height=0.0316605, viewOffsetX=0.00152443, viewOffsetY=9.62137e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.00750057, 
    height=0.0316882, viewOffsetX=0.00212562, viewOffsetY=0.000288848)
session.viewports['Viewport: 10'].view.setValues(width=0.00815682, 
    height=0.031633, viewOffsetX=-0.000284552, viewOffsetY=-9.67523e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.00747603, 
    height=0.0315845, viewOffsetX=0.00211866, viewOffsetY=0.000287905)
session.viewports['Viewport: 12'].view.setValues(width=0.0078984, 
    height=0.0316191, viewOffsetX=0.000722042, viewOffsetY=-0.000290147)
session.viewports['Viewport: 1'].view.setValues(width=0.00776859, 
    height=0.0297362, viewOffsetX=-0.000593475, viewOffsetY=-0.000111211)
session.viewports['Viewport: 2'].view.setValues(width=0.00711823, 
    height=0.0297085, viewOffsetX=0.0020439, viewOffsetY=-0.000306088)
session.viewports['Viewport: 3'].view.setValues(width=0.00721578, 
    height=0.0297951, viewOffsetX=0.00163731, viewOffsetY=8.34744e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.00705864, 
    height=0.0298211, viewOffsetX=0.00224392, viewOffsetY=0.000277946)
session.viewports['Viewport: 10'].view.setValues(width=0.00767626, 
    height=0.0297694, viewOffsetX=-0.000188175, viewOffsetY=-0.000111332)
session.viewports['Viewport: 11'].view.setValues(width=0.00703555, 
    height=0.0297236, viewOffsetX=0.00223658, viewOffsetY=0.000277038)
session.viewports['Viewport: 12'].view.setValues(width=0.00743308, 
    height=0.0297564, viewOffsetX=0.000827772, viewOffsetY=-0.000306578)
session.viewports['Viewport: 1'].view.setValues(width=0.00731041, 
    height=0.0279824, viewOffsetX=-0.000506249, viewOffsetY=-0.000124932)
session.viewports['Viewport: 2'].view.setValues(width=0.00669842, 
    height=0.0279564, viewOffsetX=0.00215378, viewOffsetY=-0.000321543)
session.viewports['Viewport: 3'].view.setValues(width=0.00679019, 
    height=0.0280378, viewOffsetX=0.00174365, viewOffsetY=7.14727e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.00664231, 
    height=0.0280622, viewOffsetX=0.00235537, viewOffsetY=0.000267674)
session.viewports['Viewport: 10'].view.setValues(width=0.00722352, 
    height=0.0280136, viewOffsetX=-9.7378e-05, viewOffsetY=-0.000125068)
session.viewports['Viewport: 11'].view.setValues(width=0.00662058, 
    height=0.0279704, viewOffsetX=0.00234767, viewOffsetY=0.000266801)
session.viewports['Viewport: 12'].view.setValues(width=0.0069947, 
    height=0.0280014, viewOffsetX=0.000927382, viewOffsetY=-0.000322058)
session.viewports['Viewport: 1'].view.setValues(width=0.00687881, 
    height=0.0263303, viewOffsetX=-0.000424084, viewOffsetY=-0.000137857)
session.viewports['Viewport: 2'].view.setValues(width=0.00630296, 
    height=0.0263059, viewOffsetX=0.00225727, viewOffsetY=-0.000336102)
session.viewports['Viewport: 3'].view.setValues(width=0.00638929, 
    height=0.0263824, viewOffsetX=0.00184382, viewOffsetY=6.01674e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.00625013, 
    height=0.0264054, viewOffsetX=0.00246035, viewOffsetY=0.000257999)
session.viewports['Viewport: 10'].view.setValues(width=0.00679705, 
    height=0.0263597, viewOffsetX=-1.18484e-05, viewOffsetY=-0.000138007)
session.viewports['Viewport: 11'].view.setValues(width=0.00622968, 
    height=0.026319, viewOffsetX=0.00245231, viewOffsetY=0.000257157)
session.viewports['Viewport: 12'].view.setValues(width=0.00658175, 
    height=0.0263483, viewOffsetX=0.00102121, viewOffsetY=-0.000336641)
session.viewports['Viewport: 1'].view.setValues(width=0.0064723, 
    height=0.0247743, viewOffsetX=-0.000344013, viewOffsetY=-0.000152706)
session.viewports['Viewport: 2'].view.setValues(width=0.00593049, 
    height=0.0247514, viewOffsetX=0.00235744, viewOffsetY=-0.000352493)
session.viewports['Viewport: 3'].view.setValues(width=0.0060117, 
    height=0.0248233, viewOffsetX=0.00194085, viewOffsetY=4.68429e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.00588076, 
    height=0.0248448, viewOffsetX=0.00256192, viewOffsetY=0.000246213)
session.viewports['Viewport: 10'].view.setValues(width=0.00639538, 
    height=0.024802, viewOffsetX=7.13946e-05, viewOffsetY=-0.000152873)
session.viewports['Viewport: 11'].view.setValues(width=0.00586152, 
    height=0.0247636, viewOffsetX=0.00255354, viewOffsetY=0.000245409)
session.viewports['Viewport: 12'].view.setValues(width=0.0061928, 
    height=0.0247912, viewOffsetX=0.00111228, viewOffsetY=-0.000353058)
session.viewports['Viewport: 1'].view.setValues(width=0.00608947, 
    height=0.0233089, viewOffsetX=-0.000359468, viewOffsetY=-0.000149068)
session.viewports['Viewport: 2'].view.setValues(width=0.00557972, 
    height=0.0232874, viewOffsetX=0.00236084, viewOffsetY=-0.000350293)
session.viewports['Viewport: 3'].view.setValues(width=0.00565611, 
    height=0.023355, viewOffsetX=0.00194134, viewOffsetY=5.19215e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0055329, 
    height=0.0233752, viewOffsetX=0.00256675, viewOffsetY=0.000252725)
session.viewports['Viewport: 10'].view.setValues(width=0.00601709, 
    height=0.023335, viewOffsetX=5.88253e-05, viewOffsetY=-0.000149231)
session.viewports['Viewport: 11'].view.setValues(width=0.0055148, 
    height=0.0232987, viewOffsetX=0.00255836, viewOffsetY=0.0002519)
session.viewports['Viewport: 12'].view.setValues(width=0.00582651, 
    height=0.0233249, viewOffsetX=0.00110696, viewOffsetY=-0.000350854)
session.viewports['Viewport: 1'].view.setValues(width=0.00572898, 
    height=0.0219291, viewOffsetX=-0.000374022, viewOffsetY=-0.000145642)
session.viewports['Viewport: 2'].view.setValues(width=0.00524941, 
    height=0.0219089, viewOffsetX=0.00236404, viewOffsetY=-0.000348222)
session.viewports['Viewport: 3'].view.setValues(width=0.00532126, 
    height=0.0219724, viewOffsetX=0.00194181, viewOffsetY=5.67038e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.00520534, 
    height=0.0219914, viewOffsetX=0.0025713, viewOffsetY=0.000258857)
session.viewports['Viewport: 10'].view.setValues(width=0.00566089, 
    height=0.0219535, viewOffsetX=4.69895e-05, viewOffsetY=-0.000145802)
session.viewports['Viewport: 11'].view.setValues(width=0.00518831, 
    height=0.0219194, viewOffsetX=0.0025629, viewOffsetY=0.000258012)
session.viewports['Viewport: 12'].view.setValues(width=0.0054816, 
    height=0.0219441, viewOffsetX=0.00110196, viewOffsetY=-0.00034878)
session.viewports['Viewport: 1'].view.setValues(width=0.00538955, 
    height=0.0206298, viewOffsetX=-0.000387725, viewOffsetY=-0.000142417)
session.viewports['Viewport: 2'].view.setValues(width=0.00493841, 
    height=0.0206109, viewOffsetX=0.00236706, viewOffsetY=-0.000346271)
session.viewports['Viewport: 3'].view.setValues(width=0.00500599, 
    height=0.0206705, viewOffsetX=0.00194224, viewOffsetY=6.12066e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.00489693, 
    height=0.0206884, viewOffsetX=0.00257559, viewOffsetY=0.000264631)
session.viewports['Viewport: 10'].view.setValues(width=0.0053255, 
    height=0.0206529, viewOffsetX=3.58454e-05, viewOffsetY=-0.000142573)
session.viewports['Viewport: 11'].view.setValues(width=0.00488091, 
    height=0.0206207, viewOffsetX=0.00256717, viewOffsetY=0.000263768)
session.viewports['Viewport: 12'].view.setValues(width=0.00515684, 
    height=0.020644, viewOffsetX=0.00109724, viewOffsetY=-0.000346826)
session.viewports['Viewport: 1'].view.setValues(width=0.00507, 
    height=0.0194067, viewOffsetX=-0.000400625, viewOffsetY=-0.00013938)
session.viewports['Viewport: 2'].view.setValues(width=0.00464561, 
    height=0.0193888, viewOffsetX=0.0023699, viewOffsetY=-0.000344435)
session.viewports['Viewport: 3'].view.setValues(width=0.00470917, 
    height=0.0194449, viewOffsetX=0.00194265, viewOffsetY=6.54459e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.00460657, 
    height=0.0194617, viewOffsetX=0.00257963, viewOffsetY=0.000270067)
session.viewports['Viewport: 10'].view.setValues(width=0.00500974, 
    height=0.0194283, viewOffsetX=2.53536e-05, viewOffsetY=-0.000139533)
session.viewports['Viewport: 11'].view.setValues(width=0.0045915, 
    height=0.019398, viewOffsetX=0.00257119, viewOffsetY=0.000269186)
session.viewports['Viewport: 12'].view.setValues(width=0.00485109, 
    height=0.01942, viewOffsetX=0.00109281, viewOffsetY=-0.000344987)
session.viewports['Viewport: 1'].view.setValues(width=0.00476918, 
    height=0.0182552, viewOffsetX=-0.00041277, viewOffsetY=-0.000136522)
session.viewports['Viewport: 2'].view.setValues(width=0.00436997, 
    height=0.0182385, viewOffsetX=0.00237257, viewOffsetY=-0.000342707)
session.viewports['Viewport: 3'].view.setValues(width=0.00442975, 
    height=0.0182912, viewOffsetX=0.00194304, viewOffsetY=6.94365e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.00433324, 
    height=0.0183069, viewOffsetX=0.00258343, viewOffsetY=0.000275184)
session.viewports['Viewport: 10'].view.setValues(width=0.0047125, 
    height=0.0182756, viewOffsetX=1.54769e-05, viewOffsetY=-0.000136671)
session.viewports['Viewport: 11'].view.setValues(width=0.00431906, 
    height=0.018247, viewOffsetX=0.00257498, viewOffsetY=0.000274286)
session.viewports['Viewport: 12'].view.setValues(width=0.00456326, 
    height=0.0182678, viewOffsetX=0.00108863, viewOffsetY=-0.000343256)
session.viewports['Viewport: 1'].view.setValues(width=0.00448602, 
    height=0.0171713, viewOffsetX=-0.000424201, viewOffsetY=-0.000133831)
session.viewports['Viewport: 2'].view.setValues(width=0.00411052, 
    height=0.0171556, viewOffsetX=0.00237509, viewOffsetY=-0.00034108)
session.viewports['Viewport: 3'].view.setValues(width=0.00416674, 
    height=0.0172052, viewOffsetX=0.00194341, viewOffsetY=7.31929e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.00407595, 
    height=0.01722, viewOffsetX=0.002587, viewOffsetY=0.000280001)
session.viewports['Viewport: 10'].view.setValues(width=0.0044327, 
    height=0.0171905, viewOffsetX=6.18004e-06, viewOffsetY=-0.000133977)
session.viewports['Viewport: 11'].view.setValues(width=0.00406262, 
    height=0.0171636, viewOffsetX=0.00257854, viewOffsetY=0.000279087)
session.viewports['Viewport: 12'].view.setValues(width=0.00429233, 
    height=0.0171832, viewOffsetX=0.0010847, viewOffsetY=-0.000341626)
session.viewports['Viewport: 1'].view.setValues(width=0.00421951, 
    height=0.0161512, viewOffsetX=-0.000434961, viewOffsetY=-0.000131298)
session.viewports['Viewport: 2'].view.setValues(width=0.00386632, 
    height=0.0161364, viewOffsetX=0.00237746, viewOffsetY=-0.000339548)
session.viewports['Viewport: 3'].view.setValues(width=0.00391919, 
    height=0.016183, viewOffsetX=0.00194375, viewOffsetY=7.67285e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.00383379, 
    height=0.0161969, viewOffsetX=0.00259037, viewOffsetY=0.000284535)
session.viewports['Viewport: 10'].view.setValues(width=0.00416936, 
    height=0.0161692, viewOffsetX=-2.57033e-06, viewOffsetY=-0.000131442)
session.viewports['Viewport: 11'].view.setValues(width=0.00382125, 
    height=0.0161439, viewOffsetX=0.0025819, viewOffsetY=0.000283606)
session.viewports['Viewport: 12'].view.setValues(width=0.00403733, 
    height=0.0161624, viewOffsetX=0.001081, viewOffsetY=-0.000340092)
session.viewports['Viewport: 1'].view.setValues(width=0.00449148, 
    height=0.0171923, viewOffsetX=-0.000512215, viewOffsetY=-0.000172393)
session.viewports['Viewport: 2'].view.setValues(width=0.00411554, 
    height=0.0171765, viewOffsetX=0.00229042, viewOffsetY=-0.000379925)
session.viewports['Viewport: 3'].view.setValues(width=0.00417181, 
    height=0.0172261, viewOffsetX=0.00185825, viewOffsetY=3.48724e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0040809, 
    height=0.0172409, viewOffsetX=0.00250269, viewOffsetY=0.000241963)
session.viewports['Viewport: 10'].view.setValues(width=0.0044381, 
    height=0.0172115, viewOffsetX=-8.14066e-05, viewOffsetY=-0.000172582)
session.viewports['Viewport: 11'].view.setValues(width=0.00406755, 
    height=0.0171845, viewOffsetX=0.00249451, viewOffsetY=0.000241174)
session.viewports['Viewport: 12'].view.setValues(width=0.00429757, 
    height=0.0172042, viewOffsetX=0.000998315, viewOffsetY=-0.000380533)
session.viewports['Viewport: 1'].view.setValues(width=0.0047753, 
    height=0.0182786, viewOffsetX=-0.000593768, viewOffsetY=-0.000215898)
session.viewports['Viewport: 2'].view.setValues(width=0.00437559, 
    height=0.0182619, viewOffsetX=0.00219501, viewOffsetY=-0.00042241)
session.viewports['Viewport: 3'].view.setValues(width=0.00443543, 
    height=0.0183146, viewOffsetX=0.00176501, viewOffsetY=-9.69757e-06)
session.viewports['Viewport: 4'].view.setValues(width=0.00433878, 
    height=0.0183303, viewOffsetX=0.00240636, viewOffsetY=0.000196379)
session.viewports['Viewport: 10'].view.setValues(width=0.00471855, 
    height=0.018299, viewOffsetX=-0.000165174, viewOffsetY=-0.000216136)
session.viewports['Viewport: 11'].view.setValues(width=0.00432458, 
    height=0.0182704, viewOffsetX=0.00239849, viewOffsetY=0.000195739)
session.viewports['Viewport: 12'].view.setValues(width=0.00456913, 
    height=0.0182913, viewOffsetX=0.000909128, viewOffsetY=-0.000423087)
session.viewports['Viewport: 1'].view.setValues(width=0.00507691, 
    height=0.0194331, viewOffsetX=-0.000680428, viewOffsetY=-0.000262125)
session.viewports['Viewport: 2'].view.setValues(width=0.00465195, 
    height=0.0194153, viewOffsetX=0.00209366, viewOffsetY=-0.000467556)
session.viewports['Viewport: 3'].view.setValues(width=0.00471558, 
    height=0.0194714, viewOffsetX=0.00166597, viewOffsetY=-5.70543e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.00461283, 
    height=0.0194881, viewOffsetX=0.00230403, viewOffsetY=0.000147946)
session.viewports['Viewport: 10'].view.setValues(width=0.00501657, 
    height=0.0194548, viewOffsetX=-0.00025418, viewOffsetY=-0.000262415)
session.viewports['Viewport: 11'].view.setValues(width=0.00459774, 
    height=0.0194244, viewOffsetX=0.00229649, viewOffsetY=0.000147465)
session.viewports['Viewport: 12'].view.setValues(width=0.00485771, 
    height=0.0194466, viewOffsetX=0.000814376, viewOffsetY=-0.000468306)
session.viewports['Viewport: 1'].view.setValues(width=0.00539736, 
    height=0.0206597, viewOffsetX=-0.000772499, viewOffsetY=-0.00031124)
session.viewports['Viewport: 2'].view.setValues(width=0.00494557, 
    height=0.0206408, viewOffsetX=0.00198598, viewOffsetY=-0.000515521)
session.viewports['Viewport: 3'].view.setValues(width=0.00501323, 
    height=0.0207004, viewOffsetX=0.00156073, viewOffsetY=-0.000107369)
session.viewports['Viewport: 4'].view.setValues(width=0.004904, 
    height=0.0207183, viewOffsetX=0.00219531, viewOffsetY=9.64887e-05)
session.viewports['Viewport: 10'].view.setValues(width=0.00533321, 
    height=0.0206828, viewOffsetX=-0.000348745, viewOffsetY=-0.000311584)
session.viewports['Viewport: 11'].view.setValues(width=0.00488795, 
    height=0.0206505, viewOffsetX=0.00218813, viewOffsetY=9.61755e-05)
session.viewports['Viewport: 12'].view.setValues(width=0.00516432, 
    height=0.020674, viewOffsetX=0.000713706, viewOffsetY=-0.000516348)
session.viewports['Viewport: 1'].view.setValues(width=0.00573779, 
    height=0.0219628, viewOffsetX=-0.000870312, viewOffsetY=-0.000363417)
session.viewports['Viewport: 2'].view.setValues(width=0.0052575, 
    height=0.0219426, viewOffsetX=0.00187159, viewOffsetY=-0.000566477)
session.viewports['Viewport: 3'].view.setValues(width=0.00532944, 
    height=0.0220061, viewOffsetX=0.00144894, viewOffsetY=-0.000160821)
session.viewports['Viewport: 4'].view.setValues(width=0.00521332, 
    height=0.0220251, viewOffsetX=0.0020798, viewOffsetY=4.18221e-05)
session.viewports['Viewport: 10'].view.setValues(width=0.0056696, 
    height=0.0219873, viewOffsetX=-0.000449207, viewOffsetY=-0.00036382)
session.viewports['Viewport: 11'].view.setValues(width=0.00519627, 
    height=0.021953, viewOffsetX=0.002073, viewOffsetY=4.16877e-05)
session.viewports['Viewport: 12'].view.setValues(width=0.00549004, 
    height=0.0219779, viewOffsetX=0.000606758, viewOffsetY=-0.000567386)
session.viewports['Viewport: 1'].view.setValues(width=0.00609942, 
    height=0.023347, viewOffsetX=-0.000974216, viewOffsetY=-0.000418843)
session.viewports['Viewport: 2'].view.setValues(width=0.00558885, 
    height=0.0233255, viewOffsetX=0.00175007, viewOffsetY=-0.000620607)
session.viewports['Viewport: 3'].view.setValues(width=0.00566534, 
    height=0.0233931, viewOffsetX=0.00133018, viewOffsetY=-0.000217602)
session.viewports['Viewport: 4'].view.setValues(width=0.00554191, 
    height=0.0234133, viewOffsetX=0.00195711, viewOffsetY=-1.62491e-05)
session.viewports['Viewport: 10'].view.setValues(width=0.00602693, 
    height=0.0233731, viewOffsetX=-0.000555925, viewOffsetY=-0.000419308)
session.viewports['Viewport: 11'].view.setValues(width=0.00552378, 
    height=0.0233367, viewOffsetX=0.00195071, viewOffsetY=-1.61934e-05)
session.viewports['Viewport: 12'].view.setValues(width=0.00583605, 
    height=0.0233631, viewOffsetX=0.000493151, viewOffsetY=-0.000621603)
session.viewports['Viewport: 1'].view.setValues(width=0.00648353, 
    height=0.0248173, viewOffsetX=-0.00112752, viewOffsetY=-0.000470157)
session.viewports['Viewport: 2'].view.setValues(width=0.0059408, 
    height=0.0247945, viewOffsetX=0.00157802, viewOffsetY=-0.000670537)
session.viewports['Viewport: 3'].view.setValues(width=0.00602212, 
    height=0.0248663, viewOffsetX=0.00116108, viewOffsetY=-0.000270352)
session.viewports['Viewport: 4'].view.setValues(width=0.00589093, 
    height=0.0248878, viewOffsetX=0.00178386, viewOffsetY=-7.03766e-05)
session.viewports['Viewport: 10'].view.setValues(width=0.00640648, 
    height=0.024845, viewOffsetX=-0.000712268, viewOffsetY=-0.000470679)
session.viewports['Viewport: 11'].view.setValues(width=0.00587166, 
    height=0.0248064, viewOffsetX=0.00177803, viewOffsetY=-7.01437e-05)
session.viewports['Viewport: 12'].view.setValues(width=0.00620357, 
    height=0.0248343, viewOffsetX=0.000329436, viewOffsetY=-0.000671613)
session.viewports['Viewport: 1'].view.setValues(width=0.00689148, 
    height=0.0263788, viewOffsetX=-0.00129034, viewOffsetY=-0.000524656)
session.viewports['Viewport: 2'].view.setValues(width=0.0063146, 
    height=0.0263545, viewOffsetX=0.0013953, viewOffsetY=-0.000723566)
session.viewports['Viewport: 3'].view.setValues(width=0.00640105, 
    height=0.026431, viewOffsetX=0.000981492, viewOffsetY=-0.000326377)
session.viewports['Viewport: 4'].view.setValues(width=0.00626162, 
    height=0.0264539, viewOffsetX=0.00159986, viewOffsetY=-0.000127864)
session.viewports['Viewport: 10'].view.setValues(width=0.00680958, 
    height=0.0264083, viewOffsetX=-0.000878315, viewOffsetY=-0.000525238)
session.viewports['Viewport: 11'].view.setValues(width=0.00624113, 
    height=0.0263673, viewOffsetX=0.00159463, viewOffsetY=-0.000127443)
session.viewports['Viewport: 12'].view.setValues(width=0.0065939, 
    height=0.0263969, viewOffsetX=0.00015556, viewOffsetY=-0.000724727)
session.viewports['Viewport: 1'].view.setValues(width=0.00732471, 
    height=0.0280371, viewOffsetX=-0.00146325, viewOffsetY=-0.000582531)
session.viewports['Viewport: 2'].view.setValues(width=0.00671155, 
    height=0.0280112, viewOffsetX=0.00120125, viewOffsetY=-0.00077988)
session.viewports['Viewport: 3'].view.setValues(width=0.00680346, 
    height=0.0280926, viewOffsetX=0.000790772, viewOffsetY=-0.000385873)
session.viewports['Viewport: 4'].view.setValues(width=0.00665527, 
    height=0.028117, viewOffsetX=0.00140446, viewOffsetY=-0.000188913)
session.viewports['Viewport: 10'].view.setValues(width=0.00723766, 
    height=0.0280684, viewOffsetX=-0.00105465, viewOffsetY=-0.000583178)
session.viewports['Viewport: 11'].view.setValues(width=0.00663349, 
    height=0.028025, viewOffsetX=0.00139987, viewOffsetY=-0.000188293)
session.viewports['Viewport: 12'].view.setValues(width=0.00700841, 
    height=0.0280563, viewOffsetX=-2.90887e-05, viewOffsetY=-0.000781132)
session.viewports['Viewport: 1'].view.setValues(width=0.00778473, 
    height=0.029798, viewOffsetX=-0.00164686, viewOffsetY=-0.000643985)
session.viewports['Viewport: 2'].view.setValues(width=0.00713305, 
    height=0.0297704, viewOffsetX=0.000995203, viewOffsetY=-0.000839676)
session.viewports['Viewport: 3'].view.setValues(width=0.00723075, 
    height=0.029857, viewOffsetX=0.000588257, viewOffsetY=-0.000449048)
session.viewports['Viewport: 4'].view.setValues(width=0.00707326, 
    height=0.0298829, viewOffsetX=0.00119698, viewOffsetY=-0.000253738)
session.viewports['Viewport: 10'].view.setValues(width=0.00769221, 
    height=0.0298312, viewOffsetX=-0.00124189, viewOffsetY=-0.000644701)
session.viewports['Viewport: 11'].view.setValues(width=0.00705012, 
    height=0.0297851, viewOffsetX=0.00119306, viewOffsetY=-0.000252905)
session.viewports['Viewport: 12'].view.setValues(width=0.00744855, 
    height=0.0298183, viewOffsetX=-0.000225155, viewOffsetY=-0.000841025)
session.viewports['Viewport: 1'].view.setValues(width=0.00827313, 
    height=0.0316675, viewOffsetX=-0.00184179, viewOffsetY=-0.000709231)
session.viewports['Viewport: 2'].view.setValues(width=0.00758055, 
    height=0.0316381, viewOffsetX=0.000776441, viewOffsetY=-0.000903162)
session.viewports['Viewport: 3'].view.setValues(width=0.00768442, 
    height=0.0317302, viewOffsetX=0.000373244, viewOffsetY=-0.000516122)
session.viewports['Viewport: 4'].view.setValues(width=0.00751706, 
    height=0.0317578, viewOffsetX=0.000976684, viewOffsetY=-0.000322564)
session.viewports['Viewport: 10'].view.setValues(width=0.00817481, 
    height=0.0317028, viewOffsetX=-0.00144068, viewOffsetY=-0.00071002)
session.viewports['Viewport: 11'].view.setValues(width=0.00749246, 
    height=0.0316539, viewOffsetX=0.000973492, viewOffsetY=-0.000321506)
session.viewports['Viewport: 12'].view.setValues(width=0.00791585, 
    height=0.031689, viewOffsetX=-0.00043332, viewOffsetY=-0.000904613)
session.viewports['Viewport: 1'].view.setValues(width=0.00879161, 
    height=0.0336521, viewOffsetX=-0.00204872, viewOffsetY=-0.000775079)
session.viewports['Viewport: 2'].view.setValues(width=0.00805562, 
    height=0.0336208, viewOffsetX=0.000544209, viewOffsetY=-0.000967138)
session.viewports['Viewport: 3'].view.setValues(width=0.00816602, 
    height=0.0337188, viewOffsetX=0.000144989, viewOffsetY=-0.00058391)
session.viewports['Viewport: 4'].view.setValues(width=0.00798818, 
    height=0.0337482, viewOffsetX=0.000742824, viewOffsetY=-0.000392214)
session.viewports['Viewport: 10'].view.setValues(width=0.00868713, 
    height=0.0336896, viewOffsetX=-0.00165172, viewOffsetY=-0.000775941)
session.viewports['Viewport: 11'].view.setValues(width=0.00796205, 
    height=0.0336378, viewOffsetX=0.000740398, viewOffsetY=-0.000390928)
session.viewports['Viewport: 12'].view.setValues(width=0.00841192, 
    height=0.0336749, viewOffsetX=-0.000654304, viewOffsetY=-0.000968692)
session.viewports['Viewport: 1'].view.setValues(width=0.00934195, 
    height=0.0357586, viewOffsetX=-0.00226837, viewOffsetY=-0.000844973)
session.viewports['Viewport: 2'].view.setValues(width=0.00855986, 
    height=0.0357253, viewOffsetX=0.000297709, viewOffsetY=-0.00103505)
session.viewports['Viewport: 3'].view.setValues(width=0.00867721, 
    height=0.0358296, viewOffsetX=-9.72911e-05, viewOffsetY=-0.000655863)
session.viewports['Viewport: 4'].view.setValues(width=0.00848826, 
    height=0.0358609, viewOffsetX=0.000494595, viewOffsetY=-0.000466144)
session.viewports['Viewport: 10'].view.setValues(width=0.00923092, 
    height=0.0357985, viewOffsetX=-0.00187572, viewOffsetY=-0.000845913)
session.viewports['Viewport: 11'].view.setValues(width=0.00846049, 
    height=0.0357436, viewOffsetX=0.000492981, viewOffsetY=-0.000464617)
session.viewports['Viewport: 12'].view.setValues(width=0.00893848, 
    height=0.0357828, viewOffsetX=-0.000888864, viewOffsetY=-0.00103671)
session.viewports['Viewport: 1'].view.setValues(width=0.00992601, 
    height=0.0379943, viewOffsetX=-0.00250149, viewOffsetY=-0.00091915)
session.viewports['Viewport: 2'].view.setValues(width=0.00909501, 
    height=0.0379588, viewOffsetX=3.61027e-05, viewOffsetY=-0.00110711)
session.viewports['Viewport: 3'].view.setValues(width=0.00921973, 
    height=0.0380698, viewOffsetX=-0.000354421, viewOffsetY=-0.000732225)
session.viewports['Viewport: 4'].view.setValues(width=0.00901899, 
    height=0.0381031, viewOffsetX=0.000231149, viewOffsetY=-0.000544606)
session.viewports['Viewport: 10'].view.setValues(width=0.00980804, 
    height=0.0380367, viewOffsetX=-0.00211345, viewOffsetY=-0.000920173)
session.viewports['Viewport: 11'].view.setValues(width=0.00898948, 
    height=0.0379785, viewOffsetX=0.000230397, viewOffsetY=-0.000542822)
session.viewports['Viewport: 12'].view.setValues(width=0.00949729, 
    height=0.0380199, viewOffsetX=-0.0011378, viewOffsetY=-0.00110889)
session.viewports['Viewport: 1'].view.setValues(width=0.0105458, 
    height=0.0403666, viewOffsetX=-0.00274885, viewOffsetY=-0.000997861)
session.viewports['Viewport: 2'].view.setValues(width=0.00966287, 
    height=0.0403288, viewOffsetX=-0.000241495, viewOffsetY=-0.00118359)
session.viewports['Viewport: 3'].view.setValues(width=0.00979542, 
    height=0.0404469, viewOffsetX=-0.000627271, viewOffsetY=-0.000813257)
session.viewports['Viewport: 4'].view.setValues(width=0.00958216, 
    height=0.0404824, viewOffsetX=-4.84036e-05, viewOffsetY=-0.000627865)
session.viewports['Viewport: 10'].view.setValues(width=0.0104204, 
    height=0.0404116, viewOffsetX=-0.00236571, viewOffsetY=-0.000998972)
session.viewports['Viewport: 11'].view.setValues(width=0.00955081, 
    height=0.04035, viewOffsetX=-4.82416e-05, viewOffsetY=-0.000625808)
session.viewports['Viewport: 12'].view.setValues(width=0.0100903, 
    height=0.0403937, viewOffsetX=-0.00140195, viewOffsetY=-0.00118549)
session.viewports['Viewport: 1'].view.setValues(width=0.0112033, 
    height=0.0428835, viewOffsetX=-0.00301129, viewOffsetY=-0.00108137)
session.viewports['Viewport: 2'].view.setValues(width=0.0102653, 
    height=0.0428433, viewOffsetX=-0.000536014, viewOffsetY=-0.00126472)
session.viewports['Viewport: 3'].view.setValues(width=0.0104062, 
    height=0.0429689, viewOffsetX=-0.000916756, viewOffsetY=-0.000899228)
session.viewports['Viewport: 4'].view.setValues(width=0.0101797, 
    height=0.0430068, viewOffsetX=-0.000345001, viewOffsetY=-0.0007162)
session.viewports['Viewport: 10'].view.setValues(width=0.0110702, 
    height=0.0429314, viewOffsetX=-0.00263335, viewOffsetY=-0.00108258)
session.viewports['Viewport: 11'].view.setValues(width=0.0101464, 
    height=0.042866, viewOffsetX=-0.000343869, viewOffsetY=-0.000713854)
session.viewports['Viewport: 12'].view.setValues(width=0.0107194, 
    height=0.0429122, viewOffsetX=-0.0016822, viewOffsetY=-0.00126675)
session.viewports['Viewport: 1'].view.setValues(width=0.0119008, 
    height=0.0455534, viewOffsetX=-0.00328968, viewOffsetY=-0.00116996)
session.viewports['Viewport: 2'].view.setValues(width=0.0109044, 
    height=0.0455105, viewOffsetX=-0.000848434, viewOffsetY=-0.00135079)
session.viewports['Viewport: 3'].view.setValues(width=0.0110541, 
    height=0.0456443, viewOffsetX=-0.00122384, viewOffsetY=-0.000990425)
session.viewports['Viewport: 4'].view.setValues(width=0.0108135, 
    height=0.0456846, viewOffsetX=-0.00065963, viewOffsetY=-0.000809905)
session.viewports['Viewport: 10'].view.setValues(width=0.0117594, 
    height=0.0456042, viewOffsetX=-0.00291726, viewOffsetY=-0.00117126)
session.viewports['Viewport: 11'].view.setValues(width=0.0107781, 
    height=0.0455351, viewOffsetX=-0.000657468, viewOffsetY=-0.000807252)
session.viewports['Viewport: 12'].view.setValues(width=0.0113867, 
    height=0.0455838, viewOffsetX=-0.00197949, viewOffsetY=-0.00135296)
session.viewports['Viewport: 1'].view.setValues(width=0.0126406, 
    height=0.048385, viewOffsetX=-0.00358494, viewOffsetY=-0.00126391)
session.viewports['Viewport: 2'].view.setValues(width=0.0115822, 
    height=0.0483394, viewOffsetX=-0.00117978, viewOffsetY=-0.00144207)
session.viewports['Viewport: 3'].view.setValues(width=0.0117413, 
    height=0.0484817, viewOffsetX=-0.00154953, viewOffsetY=-0.00108715)
session.viewports['Viewport: 4'].view.setValues(width=0.0114857, 
    height=0.0485246, viewOffsetX=-0.000993326, viewOffsetY=-0.000909289)
session.viewports['Viewport: 10'].view.setValues(width=0.0124904, 
    height=0.048439, viewOffsetX=-0.00321837, viewOffsetY=-0.00126532)
session.viewports['Viewport: 11'].view.setValues(width=0.0114482, 
    height=0.0483658, viewOffsetX=-0.000990073, viewOffsetY=-0.000906311)
session.viewports['Viewport: 12'].view.setValues(width=0.0120945, 
    height=0.0484172, viewOffsetX=-0.00229479, viewOffsetY=-0.00144439)
session.viewports['Viewport: 1'].view.setValues(width=0.013425, 
    height=0.0513877, viewOffsetX=-0.0038928, viewOffsetY=-0.00136354)
session.viewports['Viewport: 2'].view.setValues(width=0.012301, 
    height=0.0513391, viewOffsetX=-0.0015259, viewOffsetY=-0.00153886)
session.viewports['Viewport: 3'].view.setValues(width=0.01247, 
    height=0.0514905, viewOffsetX=-0.00188966, viewOffsetY=-0.00118971)
session.viewports['Viewport: 4'].view.setValues(width=0.0121986, 
    height=0.0515362, viewOffsetX=-0.00134195, viewOffsetY=-0.00101468)
session.viewports['Viewport: 10'].view.setValues(width=0.0132655, 
    height=0.051445, viewOffsetX=-0.00353243, viewOffsetY=-0.00136505)
session.viewports['Viewport: 11'].view.setValues(width=0.0121587, 
    height=0.0513676, viewOffsetX=-0.00133756, viewOffsetY=-0.00101135)
session.viewports['Viewport: 12'].view.setValues(width=0.0128451, 
    height=0.0514218, viewOffsetX=-0.00262388, viewOffsetY=-0.00154133)
session.viewports['Viewport: 1'].view.setValues(width=0.0142567, 
    height=0.054571, viewOffsetX=-0.00421919, viewOffsetY=-0.00146916)
session.viewports['Viewport: 2'].view.setValues(width=0.0130629, 
    height=0.0545193, viewOffsetX=-0.00189285, viewOffsetY=-0.00164147)
session.viewports['Viewport: 3'].view.setValues(width=0.0132425, 
    height=0.0546804, viewOffsetX=-0.00225026, viewOffsetY=-0.00129845)
session.viewports['Viewport: 4'].view.setValues(width=0.0129543, 
    height=0.0547291, viewOffsetX=-0.00171156, viewOffsetY=-0.00112641)
session.viewports['Viewport: 10'].view.setValues(width=0.0140873, 
    height=0.0546319, viewOffsetX=-0.00386538, viewOffsetY=-0.00147079)
session.viewports['Viewport: 11'].view.setValues(width=0.0129119, 
    height=0.05455, viewOffsetX=-0.00170595, viewOffsetY=-0.00112272)
session.viewports['Viewport: 12'].view.setValues(width=0.0136407, 
    height=0.054607, viewOffsetX=-0.00297278, viewOffsetY=-0.00164411)
session.viewports['Viewport: 1'].view.setValues(width=0.0173535, 
    height=0.0664249, viewOffsetX=-0.00542826, viewOffsetY=-0.0018609)
session.viewports['Viewport: 2'].view.setValues(width=0.0159004, 
    height=0.0663617, viewOffsetX=-0.00323913, viewOffsetY=-0.00202306)
session.viewports['Viewport: 3'].view.setValues(width=0.0161191, 
    height=0.0665583, viewOffsetX=-0.00357501, viewOffsetY=-0.00170076)
session.viewports['Viewport: 4'].view.setValues(width=0.0157684, 
    height=0.0666178, viewOffsetX=-0.00306669, viewOffsetY=-0.00153883)
session.viewports['Viewport: 10'].view.setValues(width=0.0171473, 
    height=0.0664991, viewOffsetX=-0.00509679, viewOffsetY=-0.00186297)
session.viewports['Viewport: 11'].view.setValues(width=0.0157168, 
    height=0.0663998, viewOffsetX=-0.00305665, viewOffsetY=-0.00153379)
session.viewports['Viewport: 12'].view.setValues(width=0.0166037, 
    height=0.0664686, viewOffsetX=-0.00425821, viewOffsetY=-0.00202631)
session.viewports['Viewport: 1'].view.setValues(width=0.016198, 
    height=0.0620016, viewOffsetX=-0.00534992, viewOffsetY=-0.00171058)
session.viewports['Viewport: 2'].view.setValues(width=0.0148414, 
    height=0.0619419, viewOffsetX=-0.0031324, viewOffsetY=-0.00187482)
session.viewports['Viewport: 3'].view.setValues(width=0.0150459, 
    height=0.0621269, viewOffsetX=-0.00347276, viewOffsetY=-0.00154819)
session.viewports['Viewport: 4'].view.setValues(width=0.0147187, 
    height=0.0621831, viewOffsetX=-0.00295802, viewOffsetY=-0.00138417)
session.viewports['Viewport: 10'].view.setValues(width=0.0160054, 
    height=0.0620709, viewOffsetX=-0.005014, viewOffsetY=-0.00171248)
session.viewports['Viewport: 11'].view.setValues(width=0.0146705, 
    height=0.0619796, viewOffsetX=-0.00294833, viewOffsetY=-0.00137964)
session.viewports['Viewport: 12'].view.setValues(width=0.0154979, 
    height=0.0620417, viewOffsetX=-0.00416437, viewOffsetY=-0.00187784)
session.viewports['Viewport: 1'].view.setValues(width=0.0152662, 
    height=0.0584351, viewOffsetX=-0.00532605, viewOffsetY=-0.0015857)
session.viewports['Viewport: 2'].view.setValues(width=0.0139878, 
    height=0.0583791, viewOffsetX=-0.00306152, viewOffsetY=-0.00175346)
session.viewports['Viewport: 3'].view.setValues(width=0.0141803, 
    height=0.0585529, viewOffsetX=-0.0034091, viewOffsetY=-0.00141968)
session.viewports['Viewport: 4'].view.setValues(width=0.0138719, 
    height=0.0586056, viewOffsetX=-0.00288356, viewOffsetY=-0.00125218)
session.viewports['Viewport: 10'].view.setValues(width=0.0150848, 
    height=0.0585003, viewOffsetX=-0.00498287, viewOffsetY=-0.00158747)
session.viewports['Viewport: 11'].view.setValues(width=0.0138265, 
    height=0.0584138, viewOffsetX=-0.00287412, viewOffsetY=-0.00124808)
session.viewports['Viewport: 12'].view.setValues(width=0.0146065, 
    height=0.0584731, viewOffsetX=-0.00411511, viewOffsetY=-0.00175628)
session.viewports['Viewport: 1'].view.setValues(width=0.0143806, 
    height=0.0550454, viewOffsetX=-0.00530157, viewOffsetY=-0.00146719)
session.viewports['Viewport: 2'].view.setValues(width=0.0131764, 
    height=0.0549928, viewOffsetX=-0.00299345, viewOffsetY=-0.00163819)
session.viewports['Viewport: 3'].view.setValues(width=0.0133577, 
    height=0.0551561, viewOffsetX=-0.00334775, viewOffsetY=-0.0012978)
session.viewports['Viewport: 4'].view.setValues(width=0.0130671, 
    height=0.0552056, viewOffsetX=-0.0028122, viewOffsetY=-0.00112706)
session.viewports['Viewport: 10'].view.setValues(width=0.0142097, 
    height=0.0551068, viewOffsetX=-0.00495166, viewOffsetY=-0.00146883)
session.viewports['Viewport: 11'].view.setValues(width=0.0130244, 
    height=0.0550249, viewOffsetX=-0.002803, viewOffsetY=-0.00112337)
session.viewports['Viewport: 12'].view.setValues(width=0.0137592, 
    height=0.0550814, viewOffsetX=-0.00406709, viewOffsetY=-0.00164082)
session.viewports['Viewport: 1'].view.setValues(width=0.013545, 
    height=0.0518467, viewOffsetX=-0.00527854, viewOffsetY=-0.00135535)
session.viewports['Viewport: 2'].view.setValues(width=0.0124107, 
    height=0.0517973, viewOffsetX=-0.00292925, viewOffsetY=-0.00152941)
session.viewports['Viewport: 3'].view.setValues(width=0.0125814, 
    height=0.0519508, viewOffsetX=-0.00328988, viewOffsetY=-0.00118277)
session.viewports['Viewport: 4'].view.setValues(width=0.0123077, 
    height=0.0519972, viewOffsetX=-0.00274489, viewOffsetY=-0.00100899)
session.viewports['Viewport: 10'].view.setValues(width=0.013384, 
    height=0.0519045, viewOffsetX=-0.00492227, viewOffsetY=-0.00135686)
session.viewports['Viewport: 11'].view.setValues(width=0.0122674, 
    height=0.051827, viewOffsetX=-0.0027359, viewOffsetY=-0.00100568)
session.viewports['Viewport: 12'].view.setValues(width=0.0129597, 
    height=0.0518807, viewOffsetX=-0.00402181, viewOffsetY=-0.00153187)
session.viewports['Viewport: 1'].view.setValues(width=0.0127564, 
    height=0.0488281, viewOffsetX=-0.00525681, viewOffsetY=-0.0012498)
session.viewports['Viewport: 2'].view.setValues(width=0.0116882, 
    height=0.0487818, viewOffsetX=-0.00286866, viewOffsetY=-0.00142676)
session.viewports['Viewport: 3'].view.setValues(width=0.0118489, 
    height=0.048926, viewOffsetX=-0.00323527, viewOffsetY=-0.00107423)
session.viewports['Viewport: 4'].view.setValues(width=0.0115911, 
    height=0.0489695, viewOffsetX=-0.00268136, viewOffsetY=-0.000897564)
session.viewports['Viewport: 10'].view.setValues(width=0.0126047, 
    height=0.0488826, viewOffsetX=-0.00489453, viewOffsetY=-0.0012512)
session.viewports['Viewport: 11'].view.setValues(width=0.0115531, 
    height=0.0488093, viewOffsetX=-0.00267259, viewOffsetY=-0.000894624)
session.viewports['Viewport: 12'].view.setValues(width=0.0122052, 
    height=0.0488603, viewOffsetX=-0.00397909, viewOffsetY=-0.00142905)
session.viewports['Viewport: 1'].view.setValues(width=0.0120123, 
    height=0.0459802, viewOffsetX=-0.0052363, viewOffsetY=-0.00115023)
session.viewports['Viewport: 2'].view.setValues(width=0.0110065, 
    height=0.0459367, viewOffsetX=-0.0028115, viewOffsetY=-0.00132991)
session.viewports['Viewport: 3'].view.setValues(width=0.0111578, 
    height=0.0460722, viewOffsetX=-0.00318375, viewOffsetY=-0.000971821)
session.viewports['Viewport: 4'].view.setValues(width=0.010915, 
    height=0.0461132, viewOffsetX=-0.00262144, viewOffsetY=-0.000792443)
session.viewports['Viewport: 10'].view.setValues(width=0.0118696, 
    height=0.0460315, viewOffsetX=-0.00486837, viewOffsetY=-0.00115151)
session.viewports['Viewport: 11'].view.setValues(width=0.0108792, 
    height=0.0459623, viewOffsetX=-0.00261285, viewOffsetY=-0.000789846)
session.viewports['Viewport: 12'].view.setValues(width=0.0114934, 
    height=0.0460107, viewOffsetX=-0.00393878, viewOffsetY=-0.00133205)
session.viewports['Viewport: 1'].view.setValues(width=0.0113106, 
    height=0.043294, viewOffsetX=-0.00521697, viewOffsetY=-0.0010563)
session.viewports['Viewport: 2'].view.setValues(width=0.0103635, 
    height=0.0432531, viewOffsetX=-0.00275759, viewOffsetY=-0.00123856)
session.viewports['Viewport: 3'].view.setValues(width=0.0105059, 
    height=0.0433805, viewOffsetX=-0.00313516, viewOffsetY=-0.000875226)
session.viewports['Viewport: 4'].view.setValues(width=0.0102772, 
    height=0.0434189, viewOffsetX=-0.00256491, viewOffsetY=-0.000693287)
session.viewports['Viewport: 10'].view.setValues(width=0.0111761, 
    height=0.0433423, viewOffsetX=-0.00484369, viewOffsetY=-0.00105748)
session.viewports['Viewport: 11'].view.setValues(width=0.0102436, 
    height=0.0432768, viewOffsetX=-0.00255651, viewOffsetY=-0.000691016)
session.viewports['Viewport: 12'].view.setValues(width=0.0108219, 
    height=0.0433228, viewOffsetX=-0.00390076, viewOffsetY=-0.00124055)
session.viewports['Viewport: 1'].view.setValues(width=0.0106487, 
    height=0.0407607, viewOffsetX=-0.00519873, viewOffsetY=-0.000967728)
session.viewports['Viewport: 2'].view.setValues(width=0.00975716, 
    height=0.0407223, viewOffsetX=-0.00270674, viewOffsetY=-0.00115241)
session.viewports['Viewport: 3'].view.setValues(width=0.00989111, 
    height=0.040842, viewOffsetX=-0.00308934, viewOffsetY=-0.000784131)
session.viewports['Viewport: 4'].view.setValues(width=0.00967581, 
    height=0.0408781, viewOffsetX=-0.0025116, viewOffsetY=-0.000599778)
session.viewports['Viewport: 10'].view.setValues(width=0.0105222, 
    height=0.0408062, viewOffsetX=-0.00482041, viewOffsetY=-0.000968806)
session.viewports['Viewport: 11'].view.setValues(width=0.00964415, 
    height=0.0407443, viewOffsetX=-0.00250338, viewOffsetY=-0.000597813)
session.viewports['Viewport: 12'].view.setValues(width=0.0101887, 
    height=0.0407879, viewOffsetX=-0.00386491, viewOffsetY=-0.00115426)
session.viewports['Viewport: 1'].view.setValues(width=0.0100247, 
    height=0.0383721, viewOffsetX=-0.00518154, viewOffsetY=-0.00088421)
session.viewports['Viewport: 2'].view.setValues(width=0.00918541, 
    height=0.0383361, viewOffsetX=-0.0026588, viewOffsetY=-0.00107118)
session.viewports['Viewport: 3'].view.setValues(width=0.00931146, 
    height=0.0384485, viewOffsetX=-0.00304613, viewOffsetY=-0.00069824)
session.viewports['Viewport: 4'].view.setValues(width=0.00910876, 
    height=0.0384824, viewOffsetX=-0.00246134, viewOffsetY=-0.000511611)
session.viewports['Viewport: 10'].view.setValues(width=0.00990557, 
    height=0.0384149, viewOffsetX=-0.00479847, viewOffsetY=-0.000885195)
session.viewports['Viewport: 11'].view.setValues(width=0.00907895, 
    height=0.0383565, viewOffsetX=-0.00245328, viewOffsetY=-0.000509934)
session.viewports['Viewport: 12'].view.setValues(width=0.00959169, 
    height=0.0383978, viewOffsetX=-0.00383111, viewOffsetY=-0.0010729)
session.viewports['Viewport: 1'].view.setValues(width=0.00943644, 
    height=0.0361203, viewOffsetX=-0.00516533, viewOffsetY=-0.000805477)
session.viewports['Viewport: 2'].view.setValues(width=0.0086464, 
    height=0.0360865, viewOffsetX=-0.00261361, viewOffsetY=-0.000994603)
session.viewports['Viewport: 3'].view.setValues(width=0.00876502, 
    height=0.0361922, viewOffsetX=-0.0030054, viewOffsetY=-0.00061727)
session.viewports['Viewport: 4'].view.setValues(width=0.00857419, 
    height=0.036224, viewOffsetX=-0.00241396, viewOffsetY=-0.000428495)
session.viewports['Viewport: 10'].view.setValues(width=0.00932428, 
    height=0.0361606, viewOffsetX=-0.00477778, viewOffsetY=-0.000806374)
session.viewports['Viewport: 11'].view.setValues(width=0.00854614, 
    height=0.0361055, viewOffsetX=-0.00240605, viewOffsetY=-0.00042709)
session.viewports['Viewport: 12'].view.setValues(width=0.00902884, 
    height=0.0361446, viewOffsetX=-0.00379924, viewOffsetY=-0.000996201)
session.viewports['Viewport: 1'].view.setValues(width=0.00888195, 
    height=0.0339979, viewOffsetX=-0.00515006, viewOffsetY=-0.000731267)
session.viewports['Viewport: 2'].view.setValues(width=0.00813836, 
    height=0.0339661, viewOffsetX=-0.00257101, viewOffsetY=-0.000922424)
session.viewports['Viewport: 3'].view.setValues(width=0.00824997, 
    height=0.0340655, viewOffsetX=-0.00296701, viewOffsetY=-0.00054095)
session.viewports['Viewport: 4'].view.setValues(width=0.00807034, 
    height=0.0340953, viewOffsetX=-0.0023693, viewOffsetY=-0.000350154)
session.viewports['Viewport: 10'].view.setValues(width=0.00877639, 
    height=0.0340358, viewOffsetX=-0.00475829, viewOffsetY=-0.00073208)
session.viewports['Viewport: 11'].view.setValues(width=0.00804393, 
    height=0.0339837, viewOffsetX=-0.00236154, viewOffsetY=-0.000349005)
session.viewports['Viewport: 12'].view.setValues(width=0.00849833, 
    height=0.0340208, viewOffsetX=-0.0037692, viewOffsetY=-0.000923906)
session.viewports['Viewport: 1'].view.setValues(width=0.0083594, 
    height=0.0319977, viewOffsetX=-0.00516339, viewOffsetY=-0.000657874)
session.viewports['Viewport: 2'].view.setValues(width=0.00765957, 
    height=0.0319679, viewOffsetX=-0.0025586, viewOffsetY=-0.000850943)
session.viewports['Viewport: 3'].view.setValues(width=0.00776458, 
    height=0.0320612, viewOffsetX=-0.00295855, viewOffsetY=-0.00046557)
session.viewports['Viewport: 4'].view.setValues(width=0.00759551, 
    height=0.0320893, viewOffsetX=-0.00235491, viewOffsetY=-0.000272871)
session.viewports['Viewport: 10'].view.setValues(width=0.00826005, 
    height=0.0320334, viewOffsetX=-0.00476767, viewOffsetY=-0.000658606)
session.viewports['Viewport: 11'].view.setValues(width=0.00757065, 
    height=0.0319843, viewOffsetX=-0.00234721, viewOffsetY=-0.000271975)
session.viewports['Viewport: 12'].view.setValues(width=0.00799836, 
    height=0.0320193, viewOffsetX=-0.00376868, viewOffsetY=-0.00085231)
session.viewports['Viewport: 1'].view.setValues(width=0.00786703, 
    height=0.030113, viewOffsetX=-0.00517594, viewOffsetY=-0.000588719)
session.viewports['Viewport: 2'].view.setValues(width=0.00720843, 
    height=0.030085, viewOffsetX=-0.00254691, viewOffsetY=-0.000783589)
session.viewports['Viewport: 3'].view.setValues(width=0.00730722, 
    height=0.0301727, viewOffsetX=-0.00295059, viewOffsetY=-0.000394542)
session.viewports['Viewport: 4'].view.setValues(width=0.00714809, 
    height=0.030199, viewOffsetX=-0.00234136, viewOffsetY=-0.000200051)
session.viewports['Viewport: 10'].view.setValues(width=0.00777352, 
    height=0.0301466, viewOffsetX=-0.0047765, viewOffsetY=-0.000589373)
session.viewports['Viewport: 11'].view.setValues(width=0.0071247, 
    height=0.0301002, viewOffsetX=-0.0023337, viewOffsetY=-0.000199393)
session.viewports['Viewport: 12'].view.setValues(width=0.00752726, 
    height=0.0301334, viewOffsetX=-0.00376819, viewOffsetY=-0.000784848)
session.viewports['Viewport: 1'].view.setValues(width=0.00740314, 
    height=0.0283373, viewOffsetX=-0.00518777, viewOffsetY=-0.000523565)
session.viewports['Viewport: 2'].view.setValues(width=0.00678339, 
    height=0.028311, viewOffsetX=-0.0025359, viewOffsetY=-0.000720133)
session.viewports['Viewport: 3'].view.setValues(width=0.00687633, 
    height=0.0283935, viewOffsetX=-0.00294309, viewOffsetY=-0.000327624)
session.viewports['Viewport: 4'].view.setValues(width=0.00672657, 
    height=0.0284182, viewOffsetX=-0.0023286, viewOffsetY=-0.000131445)
session.viewports['Viewport: 10'].view.setValues(width=0.00731515, 
    height=0.028369, viewOffsetX=-0.00478483, viewOffsetY=-0.000524147)
session.viewports['Viewport: 11'].view.setValues(width=0.00670457, 
    height=0.0283252, viewOffsetX=-0.00232097, viewOffsetY=-0.000131012)
session.viewports['Viewport: 12'].view.setValues(width=0.00708342, 
    height=0.0283566, viewOffsetX=-0.00376773, viewOffsetY=-0.000721289)
session.viewports['Viewport: 1'].view.setValues(width=0.00696615, 
    height=0.0266647, viewOffsetX=-0.00519892, viewOffsetY=-0.00046219)
session.viewports['Viewport: 2'].view.setValues(width=0.006383, height=0.02664, 
    viewOffsetX=-0.00252553, viewOffsetY=-0.000660356)
session.viewports['Viewport: 3'].view.setValues(width=0.00647043, 
    height=0.0267175, viewOffsetX=-0.00293602, viewOffsetY=-0.000264587)
session.viewports['Viewport: 4'].view.setValues(width=0.0063295, 
    height=0.0267407, viewOffsetX=-0.00231657, viewOffsetY=-6.68179e-05)
session.viewports['Viewport: 10'].view.setValues(width=0.00688336, 
    height=0.0266944, viewOffsetX=-0.00479267, viewOffsetY=-0.000462704)
session.viewports['Viewport: 11'].view.setValues(width=0.00630879, 
    height=0.0266532, viewOffsetX=-0.00230899, viewOffsetY=-6.65963e-05)
session.viewports['Viewport: 12'].view.setValues(width=0.00666532, 
    height=0.0266828, viewOffsetX=-0.00376729, viewOffsetY=-0.000661416)
session.viewports['Viewport: 1'].view.setValues(width=0.00655456, 
    height=0.0250892, viewOffsetX=-0.00520942, viewOffsetY=-0.000404382)
session.viewports['Viewport: 2'].view.setValues(width=0.00600587, 
    height=0.025066, viewOffsetX=-0.00251576, viewOffsetY=-0.000604053)
session.viewports['Viewport: 3'].view.setValues(width=0.00608812, 
    height=0.0251388, viewOffsetX=-0.00292936, viewOffsetY=-0.000205214)
session.viewports['Viewport: 4'].view.setValues(width=0.00595551, 
    height=0.0251607, viewOffsetX=-0.00230524, viewOffsetY=-5.9469e-06)
session.viewports['Viewport: 10'].view.setValues(width=0.00647666, 
    height=0.0251172, viewOffsetX=-0.00480006, viewOffsetY=-0.000404831)
session.viewports['Viewport: 11'].view.setValues(width=0.00593603, 
    height=0.0250783, viewOffsetX=-0.0022977, viewOffsetY=-5.92455e-06)
session.viewports['Viewport: 12'].view.setValues(width=0.00627152, 
    height=0.0251064, viewOffsetX=-0.00376689, viewOffsetY=-0.000605022)
session.viewports['Viewport: 1'].view.setValues(width=0.00697933, 
    height=0.0267151, viewOffsetX=-0.00520332, viewOffsetY=-0.000463065)
session.viewports['Viewport: 2'].view.setValues(width=0.00639509, 
    height=0.0266905, viewOffsetX=-0.00252487, viewOffsetY=-0.000661607)
session.viewports['Viewport: 3'].view.setValues(width=0.00648265, 
    height=0.0267679, viewOffsetX=-0.00293613, viewOffsetY=-0.000265087)
session.viewports['Viewport: 4'].view.setValues(width=0.00634144, 
    height=0.0267911, viewOffsetX=-0.0023155, viewOffsetY=-6.69438e-05)
session.viewports['Viewport: 10'].view.setValues(width=0.00689638, 
    height=0.0267449, viewOffsetX=-0.0047963, viewOffsetY=-0.000463579)
session.viewports['Viewport: 11'].view.setValues(width=0.00632069, 
    height=0.0267035, viewOffsetX=-0.00230792, viewOffsetY=-6.67219e-05)
session.viewports['Viewport: 12'].view.setValues(width=0.00667795, 
    height=0.0267334, viewOffsetX=-0.00376898, viewOffsetY=-0.000662669)
session.viewports['Viewport: 1'].view.setValues(width=0.00741781, 
    height=0.0283935, viewOffsetX=-0.00518684, viewOffsetY=-0.000524603)
session.viewports['Viewport: 2'].view.setValues(width=0.00679685, 
    height=0.0283672, viewOffsetX=-0.00252971, viewOffsetY=-0.000721562)
session.viewports['Viewport: 3'].view.setValues(width=0.00688993, 
    height=0.0284496, viewOffsetX=-0.00293769, viewOffsetY=-0.000328272)
session.viewports['Viewport: 4'].view.setValues(width=0.00673986, 
    height=0.0284743, viewOffsetX=-0.00232199, viewOffsetY=-0.000131704)
session.viewports['Viewport: 10'].view.setValues(width=0.00732964, 
    height=0.0284252, viewOffsetX=-0.00478308, viewOffsetY=-0.000525186)
session.viewports['Viewport: 11'].view.setValues(width=0.00671781, 
    height=0.0283812, viewOffsetX=-0.00231439, viewOffsetY=-0.00013127)
session.viewports['Viewport: 12'].view.setValues(width=0.00709748, 
    height=0.0284129, viewOffsetX=-0.00376396, viewOffsetY=-0.00072272)
session.viewports['Viewport: 1'].view.setValues(width=0.00788357, 
    height=0.0301763, viewOffsetX=-0.00516949, viewOffsetY=-0.000589957)
session.viewports['Viewport: 2'].view.setValues(width=0.00722362, 
    height=0.0301484, viewOffsetX=-0.00253492, viewOffsetY=-0.000785241)
session.viewports['Viewport: 3'].view.setValues(width=0.00732257, 
    height=0.0302361, viewOffsetX=-0.00293944, viewOffsetY=-0.00039537)
session.viewports['Viewport: 4'].view.setValues(width=0.00716308, 
    height=0.0302624, viewOffsetX=-0.00232894, viewOffsetY=-0.00020047)
session.viewports['Viewport: 10'].view.setValues(width=0.00778988, 
    height=0.03021, viewOffsetX=-0.00476919, viewOffsetY=-0.000590613)
session.viewports['Viewport: 11'].view.setValues(width=0.00713965, 
    height=0.0301634, viewOffsetX=-0.00232131, viewOffsetY=-0.000199811)
session.viewports['Viewport: 12'].view.setValues(width=0.00754313, 
    height=0.0301969, viewOffsetX=-0.00375874, viewOffsetY=-0.000786502)
session.viewports['Viewport: 1'].view.setValues(width=0.00837807, 
    height=0.0320691, viewOffsetX=-0.00515106, viewOffsetY=-0.000659343)
session.viewports['Viewport: 2'].view.setValues(width=0.00767671, 
    height=0.0320394, viewOffsetX=-0.00254045, viewOffsetY=-0.000852847)
session.viewports['Viewport: 3'].view.setValues(width=0.00778189, 
    height=0.0321327, viewOffsetX=-0.00294129, viewOffsetY=-0.000466608)
session.viewports['Viewport: 4'].view.setValues(width=0.00761241, 
    height=0.0321607, viewOffsetX=-0.00233631, viewOffsetY=-0.000273478)
session.viewports['Viewport: 10'].view.setValues(width=0.0082785, 
    height=0.0321049, viewOffsetX=-0.00475443, viewOffsetY=-0.000660077)
session.viewports['Viewport: 11'].view.setValues(width=0.00758751, 
    height=0.0320555, viewOffsetX=-0.00232867, viewOffsetY=-0.000272581)
session.viewports['Viewport: 12'].view.setValues(width=0.00801625, 
    height=0.0320909, viewOffsetX=-0.0037532, viewOffsetY=-0.000854217)
session.viewports['Viewport: 1'].view.setValues(width=0.00890301, 
    height=0.0340784, viewOffsetX=-0.0051315, viewOffsetY=-0.000733)
session.viewports['Viewport: 2'].view.setValues(width=0.00815768, 
    height=0.0340468, viewOffsetX=-0.00254632, viewOffsetY=-0.000924614)
session.viewports['Viewport: 3'].view.setValues(width=0.00826949, 
    height=0.0341461, viewOffsetX=-0.00294325, viewOffsetY=-0.00054223)
session.viewports['Viewport: 4'].view.setValues(width=0.0080894, 
    height=0.0341759, viewOffsetX=-0.00234414, viewOffsetY=-0.000350981)
session.viewports['Viewport: 10'].view.setValues(width=0.00879719, 
    height=0.0341165, viewOffsetX=-0.00473877, viewOffsetY=-0.000733815)
session.viewports['Viewport: 11'].view.setValues(width=0.00806294, 
    height=0.034064, viewOffsetX=-0.00233647, viewOffsetY=-0.00034983)
session.viewports['Viewport: 12'].view.setValues(width=0.0085185, 
    height=0.0341016, viewOffsetX=-0.00374731, viewOffsetY=-0.0009261)
session.viewports['Viewport: 1'].view.setValues(width=0.00946018, 
    height=0.0362111, viewOffsetX=-0.00511074, viewOffsetY=-0.00081118)
session.viewports['Viewport: 2'].view.setValues(width=0.00866819, 
    height=0.0361774, viewOffsetX=-0.00255255, viewOffsetY=-0.00100079)
session.viewports['Viewport: 3'].view.setValues(width=0.00878703, 
    height=0.0362831, viewOffsetX=-0.00294533, viewOffsetY=-0.000622497)
session.viewports['Viewport: 4'].view.setValues(width=0.00859569, 
    height=0.0363148, viewOffsetX=-0.00235246, viewOffsetY=-0.000433243)
session.viewports['Viewport: 10'].view.setValues(width=0.00934774, 
    height=0.0362516, viewOffsetX=-0.00472214, viewOffsetY=-0.000812082)
session.viewports['Viewport: 11'].view.setValues(width=0.00856757, 
    height=0.036196, viewOffsetX=-0.00234475, viewOffsetY=-0.000431823)
session.viewports['Viewport: 12'].view.setValues(width=0.00905159, 
    height=0.0362356, viewOffsetX=-0.00374106, viewOffsetY=-0.0010024)
session.viewports['Viewport: 1'].view.setValues(width=0.0100515, 
    height=0.0384745, viewOffsetX=-0.0050887, viewOffsetY=-0.000894148)
session.viewports['Viewport: 2'].view.setValues(width=0.00920997, 
    height=0.0384386, viewOffsetX=-0.00255916, viewOffsetY=-0.00108163)
session.viewports['Viewport: 3'].view.setValues(width=0.00933627, 
    height=0.038551, viewOffsetX=-0.00294755, viewOffsetY=-0.000707681)
session.viewports['Viewport: 4'].view.setValues(width=0.00913299, 
    height=0.0385848, viewOffsetX=-0.00236128, viewOffsetY=-0.000520546)
session.viewports['Viewport: 10'].view.setValues(width=0.00993201, 
    height=0.0385174, viewOffsetX=-0.00470449, viewOffsetY=-0.000895144)
session.viewports['Viewport: 11'].view.setValues(width=0.00910311, 
    height=0.0384585, viewOffsetX=-0.00235355, viewOffsetY=-0.00051884)
session.viewports['Viewport: 12'].view.setValues(width=0.00961733, 
    height=0.0385004, viewOffsetX=-0.00373443, viewOffsetY=-0.00108337)
session.viewports['Viewport: 1'].view.setValues(width=0.0106789, 
    height=0.0408761, viewOffsetX=-0.00506532, viewOffsetY=-0.000982186)
session.viewports['Viewport: 2'].view.setValues(width=0.00978484, 
    height=0.0408379, viewOffsetX=-0.00256618, viewOffsetY=-0.00116741)
session.viewports['Viewport: 3'].view.setValues(width=0.00991907, 
    height=0.0409575, viewOffsetX=-0.00294989, viewOffsetY=-0.000798069)
session.viewports['Viewport: 4'].view.setValues(width=0.00970313, 
    height=0.0409935, viewOffsetX=-0.00237063, viewOffsetY=-0.000613183)
session.viewports['Viewport: 10'].view.setValues(width=0.010552, 
    height=0.0409217, viewOffsetX=-0.00468577, viewOffsetY=-0.00098328)
session.viewports['Viewport: 11'].view.setValues(width=0.00967138, 
    height=0.0408593, viewOffsetX=-0.00236287, viewOffsetY=-0.000611173)
session.viewports['Viewport: 12'].view.setValues(width=0.0102176, 
    height=0.0409036, viewOffsetX=-0.00372739, viewOffsetY=-0.00116928)
session.viewports['Viewport: 1'].view.setValues(width=0.0113445, 
    height=0.043424, viewOffsetX=-0.00504051, viewOffsetY=-0.00107559)
session.viewports['Viewport: 2'].view.setValues(width=0.0103947, 
    height=0.0433833, viewOffsetX=-0.00257362, viewOffsetY=-0.00125841)
session.viewports['Viewport: 3'].view.setValues(width=0.0105374, 
    height=0.0435106, viewOffsetX=-0.00295238, viewOffsetY=-0.000893965)
session.viewports['Viewport: 4'].view.setValues(width=0.010308, 
    height=0.0435489, viewOffsetX=-0.00238056, viewOffsetY=-0.000711465)
session.viewports['Viewport: 10'].view.setValues(width=0.0112097, 
    height=0.0434725, viewOffsetX=-0.0046659, viewOffsetY=-0.00107679)
session.viewports['Viewport: 11'].view.setValues(width=0.0102743, 
    height=0.0434064, viewOffsetX=-0.00237277, viewOffsetY=-0.000709134)
session.viewports['Viewport: 12'].view.setValues(width=0.0108545, 
    height=0.0434531, viewOffsetX=-0.00371992, viewOffsetY=-0.00126043)
session.viewports['Viewport: 1'].view.setValues(width=0.0120506, 
    height=0.0461267, viewOffsetX=-0.0050142, viewOffsetY=-0.00117466)
session.viewports['Viewport: 2'].view.setValues(width=0.0110417, 
    height=0.0460833, viewOffsetX=-0.00258151, viewOffsetY=-0.00135494)
session.viewports['Viewport: 3'].view.setValues(width=0.0111933, 
    height=0.0462188, viewOffsetX=-0.00295502, viewOffsetY=-0.000995686)
session.viewports['Viewport: 4'].view.setValues(width=0.0109496, 
    height=0.0462596, viewOffsetX=-0.00239109, viewOffsetY=-0.000815717)
session.viewports['Viewport: 10'].view.setValues(width=0.0119074, 
    height=0.0461782, viewOffsetX=-0.00464482, viewOffsetY=-0.00117597)
session.viewports['Viewport: 11'].view.setValues(width=0.0109138, 
    height=0.0461082, viewOffsetX=-0.00238327, viewOffsetY=-0.000813045)
session.viewports['Viewport: 12'].view.setValues(width=0.0115301, 
    height=0.0461575, viewOffsetX=-0.003712, viewOffsetY=-0.00135712)
session.viewports['Viewport: 1'].view.setValues(width=0.0127994, 
    height=0.048993, viewOffsetX=-0.00498629, viewOffsetY=-0.00127974)
session.viewports['Viewport: 2'].view.setValues(width=0.0117278, 
    height=0.0489468, viewOffsetX=-0.00258988, viewOffsetY=-0.00145732)
session.viewports['Viewport: 3'].view.setValues(width=0.0118888, 
    height=0.049091, viewOffsetX=-0.00295782, viewOffsetY=-0.00110357)
session.viewports['Viewport: 4'].view.setValues(width=0.0116301, 
    height=0.0491344, viewOffsetX=-0.00240226, viewOffsetY=-0.000926283)
session.viewports['Viewport: 10'].view.setValues(width=0.0126473, 
    height=0.0490477, viewOffsetX=-0.00462247, viewOffsetY=-0.00128116)
session.viewports['Viewport: 11'].view.setValues(width=0.011592, 
    height=0.0489737, viewOffsetX=-0.0023944, viewOffsetY=-0.00092325)
session.viewports['Viewport: 12'].view.setValues(width=0.0122465, 
    height=0.0490256, viewOffsetX=-0.00370359, viewOffsetY=-0.00145966)
session.viewports['Viewport: 1'].view.setValues(width=0.0135935, 
    height=0.0520323, viewOffsetX=-0.00495669, viewOffsetY=-0.00139115)
session.viewports['Viewport: 2'].view.setValues(width=0.0124553, 
    height=0.0519831, viewOffsetX=-0.00259876, viewOffsetY=-0.00156587)
session.viewports['Viewport: 3'].view.setValues(width=0.0126264, 
    height=0.0521365, viewOffsetX=-0.00296078, viewOffsetY=-0.00121796)
session.viewports['Viewport: 4'].view.setValues(width=0.0123516, 
    height=0.0521828, viewOffsetX=-0.0024141, viewOffsetY=-0.00104352)
session.viewports['Viewport: 10'].view.setValues(width=0.0134319, 
    height=0.0520904, viewOffsetX=-0.00459876, viewOffsetY=-0.0013927)
session.viewports['Viewport: 11'].view.setValues(width=0.0123112, 
    height=0.052012, viewOffsetX=-0.0024062, viewOffsetY=-0.00104011)
session.viewports['Viewport: 12'].view.setValues(width=0.0130062, 
    height=0.0520668, viewOffsetX=-0.00369468, viewOffsetY=-0.00156839)
session.viewports['Viewport: 1'].view.setValues(width=0.012755, 
    height=0.0488227, viewOffsetX=-0.00516985, viewOffsetY=-0.00127529)
session.viewports['Viewport: 2'].view.setValues(width=0.0116869, 
    height=0.0487764, viewOffsetX=-0.00278191, viewOffsetY=-0.00145224)
session.viewports['Viewport: 3'].view.setValues(width=0.0118476, 
    height=0.0489206, viewOffsetX=-0.00314851, viewOffsetY=-0.00109974)
session.viewports['Viewport: 4'].view.setValues(width=0.0115898, 
    height=0.0489642, viewOffsetX=-0.00259474, viewOffsetY=-0.000923074)
session.viewports['Viewport: 10'].view.setValues(width=0.0126034, 
    height=0.0488772, viewOffsetX=-0.00480752, viewOffsetY=-0.00127671)
session.viewports['Viewport: 11'].view.setValues(width=0.0115519, 
    height=0.048804, viewOffsetX=-0.00258624, viewOffsetY=-0.000920051)
session.viewports['Viewport: 12'].view.setValues(width=0.0122039, 
    height=0.048855, viewOffsetX=-0.00389207, viewOffsetY=-0.00145458)
session.viewports['Viewport: 1'].view.setValues(width=0.0120124, 
    height=0.0459804, viewOffsetX=-0.00538876, viewOffsetY=-0.00117094)
session.viewports['Viewport: 2'].view.setValues(width=0.0110066, 
    height=0.0459369, viewOffsetX=-0.00296407, viewOffsetY=-0.00135064)
session.viewports['Viewport: 3'].view.setValues(width=0.0111578, 
    height=0.0460724, viewOffsetX=-0.00333625, viewOffsetY=-0.000992533)
session.viewports['Viewport: 4'].view.setValues(width=0.010915, 
    height=0.0461133, viewOffsetX=-0.00277381, viewOffsetY=-0.000813138)
session.viewports['Viewport: 10'].view.setValues(width=0.0118696, 
    height=0.0460317, viewOffsetX=-0.00502099, viewOffsetY=-0.00117224)
session.viewports['Viewport: 11'].view.setValues(width=0.0108793, 
    height=0.0459624, viewOffsetX=-0.00276473, viewOffsetY=-0.000810474)
session.viewports['Viewport: 12'].view.setValues(width=0.0114934, 
    height=0.0460109, viewOffsetX=-0.0040916, viewOffsetY=-0.00135281)
session.viewports['Viewport: 1'].view.setValues(width=0.0113106, 
    height=0.043294, viewOffsetX=-0.00559469, viewOffsetY=-0.00107237)
session.viewports['Viewport: 2'].view.setValues(width=0.0103636, 
    height=0.0432532, viewOffsetX=-0.0031356, viewOffsetY=-0.00125463)
session.viewports['Viewport: 3'].view.setValues(width=0.0105059, 
    height=0.0433805, viewOffsetX=-0.003513, viewOffsetY=-0.000891292)
session.viewports['Viewport: 4'].view.setValues(width=0.0102772, 
    height=0.0434189, viewOffsetX=-0.00294244, viewOffsetY=-0.000709341)
session.viewports['Viewport: 10'].view.setValues(width=0.0111761, 
    height=0.0433423, viewOffsetX=-0.00522183, viewOffsetY=-0.00107356)
session.viewports['Viewport: 11'].view.setValues(width=0.0102436, 
    height=0.0432768, viewOffsetX=-0.00293281, viewOffsetY=-0.000707017)
session.viewports['Viewport: 12'].view.setValues(width=0.0108219, 
    height=0.0433228, viewOffsetX=-0.00427938, viewOffsetY=-0.00125665)
session.viewports['Viewport: 1'].view.setValues(width=0.0106487, 
    height=0.0407607, viewOffsetX=-0.00578891, viewOffsetY=-0.000979412)
session.viewports['Viewport: 2'].view.setValues(width=0.00975717, 
    height=0.0407224, viewOffsetX=-0.00329737, viewOffsetY=-0.0011641)
session.viewports['Viewport: 3'].view.setValues(width=0.00989111, 
    height=0.040842, viewOffsetX=-0.00367969, viewOffsetY=-0.000795819)
session.viewports['Viewport: 4'].view.setValues(width=0.00967581, 
    height=0.0408781, viewOffsetX=-0.00310148, viewOffsetY=-0.000611457)
session.viewports['Viewport: 10'].view.setValues(width=0.0105222, 
    height=0.0408062, viewOffsetX=-0.00541125, viewOffsetY=-0.000980503)
session.viewports['Viewport: 11'].view.setValues(width=0.00964415, 
    height=0.0407443, viewOffsetX=-0.00309133, viewOffsetY=-0.000609453)
session.viewports['Viewport: 12'].view.setValues(width=0.0101887, 
    height=0.0407879, viewOffsetX=-0.00445648, viewOffsetY=-0.00116598)
session.viewports['Viewport: 1'].view.setValues(width=0.0100247, 
    height=0.0383721, viewOffsetX=-0.00588478, viewOffsetY=-0.000871046)
session.viewports['Viewport: 2'].view.setValues(width=0.00918541, 
    height=0.0383361, viewOffsetX=-0.00336257, viewOffsetY=-0.00105801)
session.viewports['Viewport: 3'].view.setValues(width=0.00931146, 
    height=0.0384485, viewOffsetX=-0.00374958, viewOffsetY=-0.000685072)
session.viewports['Viewport: 4'].view.setValues(width=0.00910876, 
    height=0.0384824, viewOffsetX=-0.00316422, viewOffsetY=-0.000498453)
session.viewports['Viewport: 10'].view.setValues(width=0.00990557, 
    height=0.0384149, viewOffsetX=-0.00550249, viewOffsetY=-0.000872016)
session.viewports['Viewport: 11'].view.setValues(width=0.00907895, 
    height=0.0383565, viewOffsetX=-0.00315387, viewOffsetY=-0.000496819)
session.viewports['Viewport: 12'].view.setValues(width=0.00959169, 
    height=0.0383978, viewOffsetX=-0.00453601, viewOffsetY=-0.00105971)
session.viewports['Viewport: 1'].view.setValues(width=0.00943644, 
    height=0.0361203, viewOffsetX=-0.00597516, viewOffsetY=-0.000768888)
session.viewports['Viewport: 2'].view.setValues(width=0.00864641, 
    height=0.0360865, viewOffsetX=-0.00342405, viewOffsetY=-0.000957986)
session.viewports['Viewport: 3'].view.setValues(width=0.00876502, 
    height=0.0361922, viewOffsetX=-0.00381547, viewOffsetY=-0.000580669)
session.viewports['Viewport: 4'].view.setValues(width=0.00857419, 
    height=0.036224, viewOffsetX=-0.00322337, viewOffsetY=-0.000391924)
session.viewports['Viewport: 10'].view.setValues(width=0.00932429, 
    height=0.0361606, viewOffsetX=-0.00558852, viewOffsetY=-0.000769744)
session.viewports['Viewport: 11'].view.setValues(width=0.00854614, 
    height=0.0361055, viewOffsetX=-0.00321282, viewOffsetY=-0.000390639)
session.viewports['Viewport: 12'].view.setValues(width=0.00902885, 
    height=0.0361446, viewOffsetX=-0.00461099, viewOffsetY=-0.000959526)
session.viewports['Viewport: 1'].view.setValues(width=0.00888195, 
    height=0.0339979, viewOffsetX=-0.00606035, viewOffsetY=-0.000672598)
session.viewports['Viewport: 2'].view.setValues(width=0.00813836, 
    height=0.0339662, viewOffsetX=-0.003482, viewOffsetY=-0.000863711)
session.viewports['Viewport: 3'].view.setValues(width=0.00824997, 
    height=0.0340655, viewOffsetX=-0.00387757, viewOffsetY=-0.000482264)
session.viewports['Viewport: 4'].view.setValues(width=0.00807034, 
    height=0.0340953, viewOffsetX=-0.00327912, viewOffsetY=-0.000291515)
session.viewports['Viewport: 10'].view.setValues(width=0.00877639, 
    height=0.0340358, viewOffsetX=-0.0056696, viewOffsetY=-0.000673346)
session.viewports['Viewport: 11'].view.setValues(width=0.00804393, 
    height=0.0339838, viewOffsetX=-0.00326839, viewOffsetY=-0.000290558)
session.viewports['Viewport: 12'].view.setValues(width=0.00849833, 
    height=0.0340208, viewOffsetX=-0.00468166, viewOffsetY=-0.000865099)
session.viewports['Viewport: 1'].view.setValues(width=0.0083594, 
    height=0.0319977, viewOffsetX=-0.00614064, viewOffsetY=-0.000581853)
session.viewports['Viewport: 2'].view.setValues(width=0.00765958, 
    height=0.0319679, viewOffsetX=-0.00353661, viewOffsetY=-0.000774865)
session.viewports['Viewport: 3'].view.setValues(width=0.00776458, 
    height=0.0320612, viewOffsetX=-0.0039361, viewOffsetY=-0.000389526)
session.viewports['Viewport: 4'].view.setValues(width=0.00759551, 
    height=0.0320893, viewOffsetX=-0.00333166, viewOffsetY=-0.00019689)
session.viewports['Viewport: 10'].view.setValues(width=0.00826005, 
    height=0.0320334, viewOffsetX=-0.00574601, viewOffsetY=-0.0005825)
session.viewports['Viewport: 11'].view.setValues(width=0.00757066, 
    height=0.0319843, viewOffsetX=-0.00332076, viewOffsetY=-0.000196243)
session.viewports['Viewport: 12'].view.setValues(width=0.00799836, 
    height=0.0320193, viewOffsetX=-0.00474826, viewOffsetY=-0.000776109)
session.viewports['Viewport: 1'].view.setValues(width=0.00786703, 
    height=0.030113, viewOffsetX=-0.00621629, viewOffsetY=-0.000496348)
session.viewports['Viewport: 2'].view.setValues(width=0.00720843, 
    height=0.030085, viewOffsetX=-0.00358806, viewOffsetY=-0.000691148)
session.viewports['Viewport: 3'].view.setValues(width=0.00730722, 
    height=0.0301727, viewOffsetX=-0.00399124, viewOffsetY=-0.000302143)
session.viewports['Viewport: 4'].view.setValues(width=0.00714809, 
    height=0.0301991, viewOffsetX=-0.00338117, viewOffsetY=-0.000107728)
session.viewports['Viewport: 10'].view.setValues(width=0.00777353, 
    height=0.0301466, viewOffsetX=-0.00581802, viewOffsetY=-0.000496899)
session.viewports['Viewport: 11'].view.setValues(width=0.00712471, 
    height=0.0301002, viewOffsetX=-0.0033701, viewOffsetY=-0.000107373)
session.viewports['Viewport: 12'].view.setValues(width=0.00752727, 
    height=0.0301334, viewOffsetX=-0.00481101, viewOffsetY=-0.000692258)
session.viewports['Viewport: 1'].view.setValues(width=0.00740314, 
    height=0.0283373, viewOffsetX=-0.00628757, viewOffsetY=-0.00041579)
session.viewports['Viewport: 2'].view.setValues(width=0.00678339, 
    height=0.0283111, viewOffsetX=-0.00363654, viewOffsetY=-0.000612275)
session.viewports['Viewport: 3'].view.setValues(width=0.00687633, 
    height=0.0283935, viewOffsetX=-0.0040432, viewOffsetY=-0.000219817)
session.viewports['Viewport: 4'].view.setValues(width=0.00672658, 
    height=0.0284182, viewOffsetX=-0.00342781, viewOffsetY=-2.37259e-05)
session.viewports['Viewport: 10'].view.setValues(width=0.00731515, 
    height=0.028369, viewOffsetX=-0.00588586, viewOffsetY=-0.000416251)
session.viewports['Viewport: 11'].view.setValues(width=0.00670457, 
    height=0.0283252, viewOffsetX=-0.00341659, viewOffsetY=-2.36458e-05)
session.viewports['Viewport: 12'].view.setValues(width=0.00708343, 
    height=0.0283566, viewOffsetX=-0.00487014, viewOffsetY=-0.000613258)
session.viewports['Viewport: 1'].view.setValues(width=0.00696615, 
    height=0.0266647, viewOffsetX=-0.00635471, viewOffsetY=-0.000339904)
session.viewports['Viewport: 2'].view.setValues(width=0.006383, height=0.02664, 
    viewOffsetX=-0.00368221, viewOffsetY=-0.000537976)
session.viewports['Viewport: 3'].view.setValues(width=0.00647043, 
    height=0.0267175, viewOffsetX=-0.00409215, viewOffsetY=-0.000142266)
session.viewports['Viewport: 4'].view.setValues(width=0.00632951, 
    height=0.0267407, viewOffsetX=-0.00347175, viewOffsetY=5.54039e-05)
session.viewports['Viewport: 10'].view.setValues(width=0.00688336, 
    height=0.0266944, viewOffsetX=-0.00594976, viewOffsetY=-0.000340281)
session.viewports['Viewport: 11'].view.setValues(width=0.0063088, 
    height=0.0266532, viewOffsetX=-0.00346039, viewOffsetY=5.5225e-05)
session.viewports['Viewport: 12'].view.setValues(width=0.00666532, 
    height=0.0266829, viewOffsetX=-0.00492584, viewOffsetY=-0.000538839)
session.viewports['Viewport: 1'].view.setValues(width=0.00655456, 
    height=0.0250892, viewOffsetX=-0.00641795, viewOffsetY=-0.000268428)
session.viewports['Viewport: 2'].view.setValues(width=0.00600588, 
    height=0.025066, viewOffsetX=-0.00372523, viewOffsetY=-0.000467994)
session.viewports['Viewport: 3'].view.setValues(width=0.00608812, 
    height=0.0251388, viewOffsetX=-0.00413825, viewOffsetY=-6.9221e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.00595551, 
    height=0.0251607, viewOffsetX=-0.00351314, viewOffsetY=0.000129935)
session.viewports['Viewport: 10'].view.setValues(width=0.00647667, 
    height=0.0251172, viewOffsetX=-0.00600995, viewOffsetY=-0.000268725)
session.viewports['Viewport: 11'].view.setValues(width=0.00593603, 
    height=0.0250783, viewOffsetX=-0.00350164, viewOffsetY=0.000129512)
session.viewports['Viewport: 12'].view.setValues(width=0.00627152, 
    height=0.0251064, viewOffsetX=-0.0049783, viewOffsetY=-0.000468745)
session.viewports['Viewport: 1'].view.setValues(width=0.00616694, 
    height=0.0236055, viewOffsetX=-0.00647751, viewOffsetY=-0.000201113)
session.viewports['Viewport: 2'].view.setValues(width=0.00565071, 
    height=0.0235837, viewOffsetX=-0.00376574, viewOffsetY=-0.000402087)
session.viewports['Viewport: 3'].view.setValues(width=0.00572807, 
    height=0.0236521, viewOffsetX=-0.00418166, viewOffsetY=-4.2934e-07)
session.viewports['Viewport: 4'].view.setValues(width=0.0056033, 
    height=0.0236726, viewOffsetX=-0.00355211, viewOffsetY=0.000200126)
session.viewports['Viewport: 10'].view.setValues(width=0.00609365, 
    height=0.0236318, viewOffsetX=-0.00606664, viewOffsetY=-0.000201335)
session.viewports['Viewport: 11'].view.setValues(width=0.00558496, 
    height=0.0235952, viewOffsetX=-0.00354049, viewOffsetY=0.000199474)
session.viewports['Viewport: 12'].view.setValues(width=0.00590064, 
    height=0.0236216, viewOffsetX=-0.00502771, viewOffsetY=-0.000402732)
session.viewports['Viewport: 1'].view.setValues(width=0.00580192, 
    height=0.0222083, viewOffsetX=-0.0065336, viewOffsetY=-0.000137725)
session.viewports['Viewport: 2'].view.setValues(width=0.00531626, 
    height=0.0221878, viewOffsetX=-0.00380389, viewOffsetY=-0.000340023)
session.viewports['Viewport: 3'].view.setValues(width=0.00538902, 
    height=0.0222522, viewOffsetX=-0.00422255, viewOffsetY=6.43498e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.00527163, 
    height=0.0222714, viewOffsetX=-0.00358881, viewOffsetY=0.000266223)
session.viewports['Viewport: 10'].view.setValues(width=0.00573297, 
    height=0.0222331, viewOffsetX=-0.00612002, viewOffsetY=-0.000137876)
session.viewports['Viewport: 11'].view.setValues(width=0.00525438, 
    height=0.0221985, viewOffsetX=-0.00357707, viewOffsetY=0.000265354)
session.viewports['Viewport: 12'].view.setValues(width=0.00555139, 
    height=0.0222235, viewOffsetX=-0.00507424, viewOffsetY=-0.000340568)
session.viewports['Viewport: 1'].view.setValues(width=0.00545823, 
    height=0.0208927, viewOffsetX=-0.00658641, viewOffsetY=-7.804e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.00500134, 
    height=0.0208735, viewOffsetX=-0.00383981, viewOffsetY=-0.000281586)
session.viewports['Viewport: 3'].view.setValues(width=0.00506978, 
    height=0.020934, viewOffsetX=-0.00426105, viewOffsetY=0.000125344)
session.viewports['Viewport: 4'].view.setValues(width=0.00495934, 
    height=0.0209521, viewOffsetX=-0.00362337, viewOffsetY=0.000328458)
session.viewports['Viewport: 10'].view.setValues(width=0.00539336, 
    height=0.0209161, viewOffsetX=-0.00617029, viewOffsetY=-7.81242e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.00494311, 
    height=0.0208835, viewOffsetX=-0.00361151, viewOffsetY=0.000327386)
session.viewports['Viewport: 12'].view.setValues(width=0.00522255, 
    height=0.0209071, viewOffsetX=-0.00511805, viewOffsetY=-0.000282037)
session.viewports['Viewport: 1'].view.setValues(width=0.00513466, 
    height=0.0196542, viewOffsetX=-0.00663614, viewOffsetY=-2.18479e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.00470486, 
    height=0.0196361, viewOffsetX=-0.00387363, viewOffsetY=-0.000226568)
session.viewports['Viewport: 3'].view.setValues(width=0.00476923, 
    height=0.0196929, viewOffsetX=-0.00429729, viewOffsetY=0.000182769)
session.viewports['Viewport: 4'].view.setValues(width=0.00466533, 
    height=0.0197099, viewOffsetX=-0.00365591, viewOffsetY=0.000387051)
session.viewports['Viewport: 10'].view.setValues(width=0.00507363, 
    height=0.0196761, viewOffsetX=-0.00621761, viewOffsetY=-2.18693e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.00465006, 
    height=0.0196454, viewOffsetX=-0.00364394, viewOffsetY=0.000385787)
session.viewports['Viewport: 12'].view.setValues(width=0.00491295, 
    height=0.0196677, viewOffsetX=-0.00515929, viewOffsetY=-0.000226931)
session.viewports['Viewport: 1'].view.setValues(width=0.00483004, 
    height=0.0184882, viewOffsetX=-0.00668294, viewOffsetY=3.1051e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.00442575, 
    height=0.0184712, viewOffsetX=-0.00390547, viewOffsetY=-0.000174774)
session.viewports['Viewport: 3'].view.setValues(width=0.00448629, 
    height=0.0185246, viewOffsetX=-0.00433141, viewOffsetY=0.000236828)
session.viewports['Viewport: 4'].view.setValues(width=0.00438855, 
    height=0.0185406, viewOffsetX=-0.00368654, viewOffsetY=0.000442209)
session.viewports['Viewport: 10'].view.setValues(width=0.00477264, 
    height=0.0185088, viewOffsetX=-0.00626216, viewOffsetY=3.10888e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.00437419, 
    height=0.0184799, viewOffsetX=-0.00367447, viewOffsetY=0.000440765)
session.viewports['Viewport: 12'].view.setValues(width=0.0046215, 
    height=0.0185009, viewOffsetX=-0.00519812, viewOffsetY=-0.000175054)
session.viewports['Viewport: 1'].view.setValues(width=0.00454331, 
    height=0.0173906, viewOffsetX=-0.006727, viewOffsetY=8.08453e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.00416302, 
    height=0.0173747, viewOffsetX=-0.00393544, viewOffsetY=-0.000126021)
session.viewports['Viewport: 3'].view.setValues(width=0.00421996, 
    height=0.0174249, viewOffsetX=-0.00436353, viewOffsetY=0.000287714)
session.viewports['Viewport: 4'].view.setValues(width=0.00412801, 
    height=0.0174399, viewOffsetX=-0.00371537, viewOffsetY=0.00049413)
session.viewports['Viewport: 10'].view.setValues(width=0.00448931, 
    height=0.0174101, viewOffsetX=-0.00630409, viewOffsetY=8.09388e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.0041145, 
    height=0.0173828, viewOffsetX=-0.00370321, viewOffsetY=0.000492516)
session.viewports['Viewport: 12'].view.setValues(width=0.00434715, 
    height=0.0174027, viewOffsetX=-0.00523467, viewOffsetY=-0.000126221)
session.viewports['Viewport: 1'].view.setValues(width=0.00427343, 
    height=0.0163576, viewOffsetX=-0.00676848, viewOffsetY=0.000127713)
session.viewports['Viewport: 2'].view.setValues(width=0.00391573, 
    height=0.0163426, viewOffsetX=-0.00396365, viewOffsetY=-8.01316e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.00396928, 
    height=0.0163898, viewOffsetX=-0.00439376, viewOffsetY=0.00033561)
session.viewports['Viewport: 4'].view.setValues(width=0.00388279, 
    height=0.0164039, viewOffsetX=-0.00374251, viewOffsetY=0.000543)
session.viewports['Viewport: 10'].view.setValues(width=0.00422264, 
    height=0.0163759, viewOffsetX=-0.00634357, viewOffsetY=0.000127859)
session.viewports['Viewport: 11'].view.setValues(width=0.00387008, 
    height=0.0163502, viewOffsetX=-0.00373026, viewOffsetY=0.000541226)
session.viewports['Viewport: 12'].view.setValues(width=0.00408892, 
    height=0.0163689, viewOffsetX=-0.00526908, viewOffsetY=-8.02585e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.00401942, 
    height=0.0153853, viewOffsetX=-0.00680751, viewOffsetY=0.000171824)
session.viewports['Viewport: 2'].view.setValues(width=0.00368299, 
    height=0.0153713, viewOffsetX=-0.0039902, viewOffsetY=-3.69428e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.00373335, 
    height=0.0154156, viewOffsetX=-0.00442221, viewOffsetY=0.000380688)
session.viewports['Viewport: 4'].view.setValues(width=0.003652, 
    height=0.0154288, viewOffsetX=-0.00376805, viewOffsetY=0.000588993)
session.viewports['Viewport: 10'].view.setValues(width=0.00397165, 
    height=0.0154025, viewOffsetX=-0.00638071, viewOffsetY=0.000172019)
session.viewports['Viewport: 11'].view.setValues(width=0.00364005, 
    height=0.0153784, viewOffsetX=-0.00375572, viewOffsetY=0.000587069)
session.viewports['Viewport: 12'].view.setValues(width=0.00384589, 
    height=0.015396, viewOffsetX=-0.00530146, viewOffsetY=-3.70002e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.00378038, 
    height=0.0144703, viewOffsetX=-0.00684424, viewOffsetY=0.000213336)
session.viewports['Viewport: 2'].view.setValues(width=0.00346396, 
    height=0.0144571, viewOffsetX=-0.00401518, viewOffsetY=3.70195e-06)
session.viewports['Viewport: 3'].view.setValues(width=0.00351132, 
    height=0.0144988, viewOffsetX=-0.00444899, viewOffsetY=0.00042311)
session.viewports['Viewport: 4'].view.setValues(width=0.0034348, 
    height=0.0145112, viewOffsetX=-0.00379208, viewOffsetY=0.000632278)
session.viewports['Viewport: 10'].view.setValues(width=0.00373545, 
    height=0.0144865, viewOffsetX=-0.00641568, viewOffsetY=0.000213577)
session.viewports['Viewport: 11'].view.setValues(width=0.00342356, 
    height=0.0144638, viewOffsetX=-0.00377967, viewOffsetY=0.000630212)
session.viewports['Viewport: 12'].view.setValues(width=0.00361717, 
    height=0.0144804, viewOffsetX=-0.00533193, viewOffsetY=3.70998e-06)
session.viewports['Viewport: 1'].view.setValues(width=0.00355544, 
    height=0.0136093, viewOffsetX=-0.00687881, viewOffsetY=0.000252399)
session.viewports['Viewport: 2'].view.setValues(width=0.00325785, 
    height=0.0135969, viewOffsetX=-0.00403869, viewOffsetY=4.19499e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.00330238, 
    height=0.0136361, viewOffsetX=-0.00447418, viewOffsetY=0.00046303)
session.viewports['Viewport: 4'].view.setValues(width=0.00323041, 
    height=0.0136478, viewOffsetX=-0.0038147, viewOffsetY=0.000673009)
session.viewports['Viewport: 10'].view.setValues(width=0.00351318, 
    height=0.0136245, viewOffsetX=-0.00644858, viewOffsetY=0.000252684)
session.viewports['Viewport: 11'].view.setValues(width=0.00321985, 
    height=0.0136031, viewOffsetX=-0.00380222, viewOffsetY=0.00067081)
session.viewports['Viewport: 12'].view.setValues(width=0.00340194, 
    height=0.0136188, viewOffsetX=-0.00536061, viewOffsetY=4.20194e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.00334378, 
    height=0.0127991, viewOffsetX=-0.00691134, viewOffsetY=0.000289157)
session.viewports['Viewport: 2'].view.setValues(width=0.0030639, 
    height=0.0127875, viewOffsetX=-0.00406082, viewOffsetY=7.79398e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.00310578, 
    height=0.0128243, viewOffsetX=-0.00449789, viewOffsetY=0.000500593)
session.viewports['Viewport: 4'].view.setValues(width=0.00303809, 
    height=0.0128352, viewOffsetX=-0.00383598, viewOffsetY=0.000711336)
session.viewports['Viewport: 10'].view.setValues(width=0.00330403, 
    height=0.0128134, viewOffsetX=-0.00647953, viewOffsetY=0.000289483)
session.viewports['Viewport: 11'].view.setValues(width=0.00302816, 
    height=0.0127933, viewOffsetX=-0.00382343, viewOffsetY=0.000709012)
session.viewports['Viewport: 12'].view.setValues(width=0.00319942, 
    height=0.012808, viewOffsetX=-0.00538759, viewOffsetY=7.80672e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.00314462, 
    height=0.0120368, viewOffsetX=-0.00694194, viewOffsetY=0.000323743)
session.viewports['Viewport: 2'].view.setValues(width=0.00288142, 
    height=0.0120258, viewOffsetX=-0.00408164, viewOffsetY=0.000111803)
session.viewports['Viewport: 3'].view.setValues(width=0.0029208, 
    height=0.0120604, viewOffsetX=-0.0045202, viewOffsetY=0.000535937)
session.viewports['Viewport: 4'].view.setValues(width=0.00285714, 
    height=0.0120708, viewOffsetX=-0.00385601, viewOffsetY=0.000747398)
session.viewports['Viewport: 10'].view.setValues(width=0.00310725, 
    height=0.0120502, viewOffsetX=-0.00650866, viewOffsetY=0.000324107)
session.viewports['Viewport: 11'].view.setValues(width=0.00284779, 
    height=0.0120313, viewOffsetX=-0.00384339, viewOffsetY=0.000744955)
session.viewports['Viewport: 12'].view.setValues(width=0.00300886, 
    height=0.0120452, viewOffsetX=-0.00541298, viewOffsetY=0.000111985)
session.viewports['Viewport: 1'].view.setValues(width=0.00295724, 
    height=0.0113196, viewOffsetX=-0.00697073, viewOffsetY=0.000356283)
session.viewports['Viewport: 2'].view.setValues(width=0.00270973, 
    height=0.0113093, viewOffsetX=-0.00410122, viewOffsetY=0.000143664)
session.viewports['Viewport: 3'].view.setValues(width=0.00274675, 
    height=0.0113418, viewOffsetX=-0.00454119, viewOffsetY=0.00056919)
session.viewports['Viewport: 4'].view.setValues(width=0.00268689, 
    height=0.0113515, viewOffsetX=-0.00387485, viewOffsetY=0.000781327)
session.viewports['Viewport: 10'].view.setValues(width=0.0029221, 
    height=0.0113322, viewOffsetX=-0.00653607, viewOffsetY=0.000356684)
session.viewports['Viewport: 11'].view.setValues(width=0.0026781, 
    height=0.0113144, viewOffsetX=-0.00386217, viewOffsetY=0.000778773)
session.viewports['Viewport: 12'].view.setValues(width=0.00282958, 
    height=0.0113275, viewOffsetX=-0.00543686, viewOffsetY=0.000143896)
session.viewports['Viewport: 1'].view.setValues(width=0.00278096, 
    height=0.0106448, viewOffsetX=-0.00699782, viewOffsetY=0.000386896)
session.viewports['Viewport: 2'].view.setValues(width=0.0025482, 
    height=0.0106351, viewOffsetX=-0.00411965, viewOffsetY=0.000173638)
session.viewports['Viewport: 3'].view.setValues(width=0.00258302, 
    height=0.0106657, viewOffsetX=-0.00456094, viewOffsetY=0.000600475)
session.viewports['Viewport: 4'].view.setValues(width=0.00252672, 
    height=0.0106748, viewOffsetX=-0.00389258, viewOffsetY=0.000813247)
session.viewports['Viewport: 10'].view.setValues(width=0.00274791, 
    height=0.0106567, viewOffsetX=-0.00656185, viewOffsetY=0.000387331)
session.viewports['Viewport: 11'].view.setValues(width=0.00251845, 
    height=0.0106399, viewOffsetX=-0.00387984, viewOffsetY=0.000810589)
session.viewports['Viewport: 12'].view.setValues(width=0.00266091, 
    height=0.0106522, viewOffsetX=-0.00545934, viewOffsetY=0.000173919)
session.viewports['Viewport: 1'].view.setValues(width=0.00316141, 
    height=0.0121011, viewOffsetY=0.000336677)
session.viewports['Viewport: 2'].view.setValues(width=0.00289681, 
    height=0.0120901, viewOffsetX=-0.00413728, viewOffsetY=0.000124526)
session.viewports['Viewport: 3'].view.setValues(width=0.00293639, 
    height=0.0121248, viewOffsetX=-0.00457626, viewOffsetY=0.000549096)
session.viewports['Viewport: 4'].view.setValues(width=0.00287238, 
    height=0.0121352, viewOffsetX=-0.00391132, viewOffsetY=0.000760766)
session.viewports['Viewport: 10'].view.setValues(width=0.00312384, 
    height=0.0121146, viewOffsetX=-0.00656687, viewOffsetY=0.000337056)
session.viewports['Viewport: 11'].view.setValues(width=0.00286299, 
    height=0.0120955, viewOffsetX=-0.00389852, viewOffsetY=0.000758279)
session.viewports['Viewport: 12'].view.setValues(width=0.00302494, 
    height=0.0121095, viewOffsetX=-0.00547011, viewOffsetY=0.000124728)
session.viewports['Viewport: 1'].view.setValues(width=0.00358949, 
    height=0.0137397, viewOffsetX=-0.00705061, viewOffsetY=0.000293153)
session.viewports['Viewport: 2'].view.setValues(width=0.00328906, 
    height=0.0137272, viewOffsetX=-0.00420798, viewOffsetY=8.25415e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.003334, 
    height=0.0137667, viewOffsetX=-0.00464377, viewOffsetY=0.000503987)
session.viewports['Viewport: 4'].view.setValues(width=0.00326134, 
    height=0.0137784, viewOffsetX=-0.00398353, viewOffsetY=0.000714124)
session.viewports['Viewport: 10'].view.setValues(width=0.00354683, 
    height=0.013755, viewOffsetX=-0.00662016, viewOffsetY=0.000293483)
session.viewports['Viewport: 11'].view.setValues(width=0.00325067, 
    height=0.0137333, viewOffsetX=-0.00397049, viewOffsetY=0.00071179)
session.viewports['Viewport: 12'].view.setValues(width=0.00343454, 
    height=0.0137493, viewOffsetX=-0.00553139, viewOffsetY=8.26762e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.003815, 
    height=0.0146029, viewOffsetX=-0.00707516, viewOffsetY=0.000271379)
session.viewports['Viewport: 2'].view.setValues(width=0.00349569, 
    height=0.0145896, viewOffsetX=-0.0042449, viewOffsetY=6.16917e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.00354347, 
    height=0.0146316, viewOffsetX=-0.00467879, viewOffsetY=0.000481267)
session.viewports['Viewport: 4'].view.setValues(width=0.00346624, 
    height=0.0146441, viewOffsetX=-0.00402137, viewOffsetY=0.000690485)
session.viewports['Viewport: 10'].view.setValues(width=0.00376966, 
    height=0.0146192, viewOffsetX=-0.00664664, viewOffsetY=0.000271685)
session.viewports['Viewport: 11'].view.setValues(width=0.0034549, 
    height=0.0145962, viewOffsetX=-0.00400821, viewOffsetY=0.000688228)
session.viewports['Viewport: 12'].view.setValues(width=0.0036503, 
    height=0.014613, viewOffsetX=-0.00556266, viewOffsetY=6.17929e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.00405649, 
    height=0.0155272, viewOffsetX=-0.00710482, viewOffsetY=0.000248387)
session.viewports['Viewport: 2'].view.setValues(width=0.00371696, 
    height=0.0155131, viewOffsetX=-0.00428628, viewOffsetY=3.95745e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.00376777, 
    height=0.0155577, viewOffsetX=-0.00471835, viewOffsetY=0.000457376)
session.viewports['Viewport: 4'].view.setValues(width=0.00368566, 
    height=0.0155711, viewOffsetX=-0.0040636, viewOffsetY=0.000665721)
session.viewports['Viewport: 10'].view.setValues(width=0.00400828, 
    height=0.0155445, viewOffsetX=-0.00667813, viewOffsetY=0.000248667)
session.viewports['Viewport: 11'].view.setValues(width=0.0036736, 
    height=0.0155201, viewOffsetX=-0.0040503, viewOffsetY=0.000663546)
session.viewports['Viewport: 12'].view.setValues(width=0.00388136, 
    height=0.015538, viewOffsetX=-0.0055987, viewOffsetY=3.96401e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.00431311, 
    height=0.0165095, viewOffsetX=-0.00707582, viewOffsetY=0.000227303)
session.viewports['Viewport: 2'].view.setValues(width=0.0039521, 
    height=0.0164944, viewOffsetX=-0.00426969, viewOffsetY=1.94246e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.00400613, 
    height=0.016542, viewOffsetX=-0.00469987, viewOffsetY=0.000435337)
session.viewports['Viewport: 4'].view.setValues(width=0.00391883, 
    height=0.0165562, viewOffsetX=-0.004048, viewOffsetY=0.000642752)
session.viewports['Viewport: 10'].view.setValues(width=0.00426185, 
    height=0.0165279, viewOffsetX=-0.00665102, viewOffsetY=0.00022756)
session.viewports['Viewport: 11'].view.setValues(width=0.00390601, 
    height=0.016502, viewOffsetX=-0.00403475, viewOffsetY=0.000640652)
session.viewports['Viewport: 12'].view.setValues(width=0.0041269, 
    height=0.0165209, viewOffsetX=-0.00557633, viewOffsetY=1.94577e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.00458581, 
    height=0.0175533, viewOffsetX=-0.007045, viewOffsetY=0.000204898)
session.viewports['Viewport: 2'].view.setValues(width=0.00420197, 
    height=0.0175373, viewOffsetX=-0.00425206, viewOffsetY=-1.98791e-06)
session.viewports['Viewport: 3'].view.setValues(width=0.00425942, 
    height=0.0175878, viewOffsetX=-0.00468023, viewOffsetY=0.000411917)
session.viewports['Viewport: 4'].view.setValues(width=0.00416661, 
    height=0.017603, viewOffsetX=-0.00403142, viewOffsetY=0.000618344)
session.viewports['Viewport: 10'].view.setValues(width=0.00453131, 
    height=0.0175729, viewOffsetX=-0.0066222, viewOffsetY=0.00020513)
session.viewports['Viewport: 11'].view.setValues(width=0.00415298, 
    height=0.0175454, viewOffsetX=-0.00401823, viewOffsetY=0.000616324)
session.viewports['Viewport: 12'].view.setValues(width=0.00438782, 
    height=0.0175655, viewOffsetX=-0.00555256, viewOffsetY=-1.98916e-06)
session.viewports['Viewport: 1'].view.setValues(width=0.00487557, 
    height=0.0186625, viewOffsetX=-0.00701225, viewOffsetY=0.000181091)
session.viewports['Viewport: 2'].view.setValues(width=0.00446748, 
    height=0.0186454, viewOffsetX=-0.00423333, viewOffsetY=-2.47404e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.00452857, 
    height=0.0186992, viewOffsetX=-0.00465935, viewOffsetY=0.000387031)
session.viewports['Viewport: 4'].view.setValues(width=0.00442989, 
    height=0.0187153, viewOffsetX=-0.00401381, viewOffsetY=0.000592408)
session.viewports['Viewport: 10'].view.setValues(width=0.00481763, 
    height=0.0186833, viewOffsetX=-0.00659157, viewOffsetY=0.000181296)
session.viewports['Viewport: 11'].view.setValues(width=0.0044154, 
    height=0.018654, viewOffsetX=-0.00400067, viewOffsetY=0.000590473)
session.viewports['Viewport: 12'].view.setValues(width=0.00466507, 
    height=0.0186754, viewOffsetX=-0.00552731, viewOffsetY=-2.47783e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.00518345, 
    height=0.0198409, viewOffsetX=-0.00697746, viewOffsetY=0.000155796)
session.viewports['Viewport: 2'].view.setValues(width=0.00474958, 
    height=0.0198228, viewOffsetX=-0.00421343, viewOffsetY=-4.8915e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.00481454, 
    height=0.01988, viewOffsetX=-0.00463718, viewOffsetY=0.00036059)
session.viewports['Viewport: 4'].view.setValues(width=0.00470964, 
    height=0.0198971, viewOffsetX=-0.00399509, viewOffsetY=0.000564851)
session.viewports['Viewport: 10'].view.setValues(width=0.00512185, 
    height=0.0198631, viewOffsetX=-0.00655904, viewOffsetY=0.000155973)
session.viewports['Viewport: 11'].view.setValues(width=0.00469423, 
    height=0.019832, viewOffsetX=-0.00398202, viewOffsetY=0.000563006)
session.viewports['Viewport: 12'].view.setValues(width=0.00495965, 
    height=0.0198546, viewOffsetX=-0.00550047, viewOffsetY=-4.89918e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.00551054, 
    height=0.021093, viewOffsetX=-0.00693834, viewOffsetY=0.000131063)
session.viewports['Viewport: 2'].view.setValues(width=0.00504929, 
    height=0.0210736, viewOffsetX=-0.00419014, viewOffsetY=-7.24554e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.00511836, 
    height=0.0211345, viewOffsetX=-0.00461147, viewOffsetY=0.00033464)
session.viewports['Viewport: 4'].view.setValues(width=0.00500684, 
    height=0.0211528, viewOffsetX=-0.00397306, viewOffsetY=0.000537713)
session.viewports['Viewport: 10'].view.setValues(width=0.00544505, 
    height=0.0211165, viewOffsetX=-0.00652232, viewOffsetY=0.000131212)
session.viewports['Viewport: 11'].view.setValues(width=0.00499046, 
    height=0.0210836, viewOffsetX=-0.00396006, viewOffsetY=0.000535957)
session.viewports['Viewport: 12'].view.setValues(width=0.00527262, 
    height=0.0211075, viewOffsetX=-0.00546981, viewOffsetY=-7.25701e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.00585803, 
    height=0.022423, viewOffsetX=-0.00689679, viewOffsetY=0.000104789)
session.viewports['Viewport: 2'].view.setValues(width=0.00536768, 
    height=0.0224025, viewOffsetX=-0.00416539, viewOffsetY=-9.74632e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.00544112, 
    height=0.0224673, viewOffsetX=-0.00458416, viewOffsetY=0.000307072)
session.viewports['Viewport: 4'].view.setValues(width=0.00532258, 
    height=0.0224867, viewOffsetX=-0.00394966, viewOffsetY=0.000508885)
session.viewports['Viewport: 10'].view.setValues(width=0.00578841, 
    height=0.0224481, viewOffsetX=-0.00648331, viewOffsetY=0.000104909)
session.viewports['Viewport: 11'].view.setValues(width=0.00530516, 
    height=0.0224131, viewOffsetX=-0.00393673, viewOffsetY=0.000507222)
session.viewports['Viewport: 12'].view.setValues(width=0.00560509, 
    height=0.0224385, viewOffsetX=-0.00543724, viewOffsetY=-9.7618e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.00622714, 
    height=0.0238359, viewOffsetX=-0.00685265, viewOffsetY=7.68794e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.00570588, 
    height=0.023814, viewOffsetX=-0.00413911, viewOffsetY=-0.000124027)
session.viewports['Viewport: 3'].view.setValues(width=0.00578397, 
    height=0.023883, viewOffsetX=-0.00455514, viewOffsetY=0.000277789)
session.viewports['Viewport: 4'].view.setValues(width=0.00565796, 
    height=0.0239036, viewOffsetX=-0.0039248, viewOffsetY=0.000478261)
session.viewports['Viewport: 10'].view.setValues(width=0.00615313, 
    height=0.0238625, viewOffsetX=-0.00644188, viewOffsetY=7.69681e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.00563945, 
    height=0.0238254, viewOffsetX=-0.00391195, viewOffsetY=0.000476699)
session.viewports['Viewport: 12'].view.setValues(width=0.00595825, 
    height=0.0238523, viewOffsetX=-0.00540264, viewOffsetY=-0.000124225)
session.viewports['Viewport: 1'].view.setValues(width=0.00707049, 
    height=0.027064, viewOffsetX=-0.00675888, viewOffsetY=1.36212e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.00647863, 
    height=0.0270391, viewOffsetX=-0.00408333, viewOffsetY=-0.00018442)
session.viewports['Viewport: 3'].view.setValues(width=0.00656731, 
    height=0.0271175, viewOffsetX=-0.00449355, viewOffsetY=0.0002116)
session.viewports['Viewport: 4'].view.setValues(width=0.00642425, 
    height=0.027141, viewOffsetX=-0.00387203, viewOffsetY=0.000409217)
session.viewports['Viewport: 10'].view.setValues(width=0.00698646, 
    height=0.0270943, viewOffsetX=-0.00635386, viewOffsetY=1.36392e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.00640323, 
    height=0.0270522, viewOffsetX=-0.00385936, viewOffsetY=0.000407881)
session.viewports['Viewport: 12'].view.setValues(width=0.00676518, 
    height=0.0270826, viewOffsetX=-0.00532917, viewOffsetY=-0.000184714)
session.viewports['Viewport: 1'].view.setValues(width=0.00750769, 
    height=0.0287375, viewOffsetX=-0.0066994, viewOffsetY=-1.99557e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.00687922, 
    height=0.028711, viewOffsetX=-0.00404785, viewOffsetY=-0.000216191)
session.viewports['Viewport: 3'].view.setValues(width=0.00697342, 
    height=0.0287944, viewOffsetX=-0.00445442, viewOffsetY=0.000176184)
session.viewports['Viewport: 4'].view.setValues(width=0.00682154, 
    height=0.0288194, viewOffsetX=-0.00383848, viewOffsetY=0.000372002)
session.viewports['Viewport: 10'].view.setValues(width=0.00741846, 
    height=0.0287696, viewOffsetX=-0.00629802, viewOffsetY=-1.99752e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.00679922, 
    height=0.0287251, viewOffsetX=-0.00382592, viewOffsetY=0.000370788)
session.viewports['Viewport: 12'].view.setValues(width=0.00718348, 
    height=0.0287572, viewOffsetX=-0.00528251, viewOffsetY=-0.000216536)
session.viewports['Viewport: 1'].view.setValues(width=0.00852316, 
    height=0.0326245, viewOffsetX=-0.00658662, viewOffsetY=-9.61152e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.00780966, 
    height=0.0325943, viewOffsetX=-0.00398076, viewOffsetY=-0.000288902)
session.viewports['Viewport: 3'].view.setValues(width=0.00791664, 
    height=0.0326891, viewOffsetX=-0.00438033, viewOffsetY=9.64982e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.00774422, 
    height=0.0327175, viewOffsetX=-0.00377502, viewOffsetY=0.000288882)
session.viewports['Viewport: 10'].view.setValues(width=0.00842186, 
    height=0.0326609, viewOffsetX=-0.00619215, viewOffsetY=-9.62198e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.00771888, 
    height=0.0326105, viewOffsetX=-0.00376266, viewOffsetY=0.00028794)
session.viewports['Viewport: 12'].view.setValues(width=0.00815509, 
    height=0.0326467, viewOffsetX=-0.00519414, viewOffsetY=-0.000289365)
session.viewports['Viewport: 1'].view.setValues(width=0.00904666, 
    height=0.0346283, viewOffsetX=-0.00651531, viewOffsetY=-0.000136326)
session.viewports['Viewport: 2'].view.setValues(width=0.00828931, 
    height=0.0345961, viewOffsetX=-0.00393822, viewOffsetY=-0.000326948)
session.viewports['Viewport: 3'].view.setValues(width=0.00840292, 
    height=0.034697, viewOffsetX=-0.00433341, viewOffsetY=5.40809e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.00821994, 
    height=0.0347273, viewOffsetX=-0.00373479, viewOffsetY=0.000244309)
session.viewports['Viewport: 10'].view.setValues(width=0.00893914, 
    height=0.034667, viewOffsetX=-0.0061252, viewOffsetY=-0.000136476)
session.viewports['Viewport: 11'].view.setValues(width=0.00819304, 
    height=0.0346137, viewOffsetX=-0.00372257, viewOffsetY=0.000243512)
session.viewports['Viewport: 12'].view.setValues(width=0.00865595, 
    height=0.0346518, viewOffsetX=-0.00513819, viewOffsetY=-0.000327472)
session.viewports['Viewport: 1'].view.setValues(width=0.0110196, 
    height=0.0421802, viewOffsetX=-0.0063016, viewOffsetY=-0.000283905)
session.viewports['Viewport: 2'].view.setValues(width=0.0100971, 
    height=0.042141, viewOffsetX=-0.00381115, viewOffsetY=-0.000467987)
session.viewports['Viewport: 3'].view.setValues(width=0.0102355, 
    height=0.042264, viewOffsetX=-0.00419307, viewOffsetY=-0.000100189)
session.viewports['Viewport: 4'].view.setValues(width=0.0100126, 
    height=0.042301, viewOffsetX=-0.00361458, viewOffsetY=8.35224e-05)
session.viewports['Viewport: 10'].view.setValues(width=0.0108886, 
    height=0.0422273, viewOffsetX=-0.00592462, viewOffsetY=-0.000284219)
session.viewports['Viewport: 11'].view.setValues(width=0.00997986, 
    height=0.0421626, viewOffsetX=-0.00360275, viewOffsetY=8.32515e-05)
session.viewports['Viewport: 12'].view.setValues(width=0.0105437, 
    height=0.0422087, viewOffsetX=-0.00497079, viewOffsetY=-0.000468738)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.491494, 
    farPlane=0.516342, width=0.010971, height=0.041994, 
    viewOffsetX=-0.00253899, viewOffsetY=0.000357935)
session.viewports['Viewport: 2'].view.setValues(nearPlane=0.491031, 
    farPlane=0.515896, width=0.0100524, height=0.0419546, 
    viewOffsetX=-0.000372196, viewOffsetY=0.000174068)
session.viewports['Viewport: 3'].view.setValues(nearPlane=0.492537, 
    farPlane=0.517391, width=0.0101904, height=0.0420778, 
    viewOffsetX=-0.000705519, viewOffsetY=0.000542118)
session.viewports['Viewport: 4'].view.setValues(nearPlane=0.49293, 
    farPlane=0.517808, width=0.00996857, height=0.0421149, 
    viewOffsetX=-0.000205116, viewOffsetY=0.000725586)
session.viewports['Viewport: 10'].view.setValues(nearPlane=0.492039, 
    farPlane=0.516914, width=0.0108406, height=0.0420409, 
    viewOffsetX=-0.00220806, viewOffsetY=0.000358338)
session.viewports['Viewport: 11'].view.setValues(nearPlane=0.491294, 
    farPlane=0.516048, width=0.00993595, height=0.0419771, 
    viewOffsetX=-0.000204441, viewOffsetY=0.000723214)
session.viewports['Viewport: 12'].view.setValues(nearPlane=0.491861, 
    farPlane=0.516765, width=0.010497, height=0.0420221, 
    viewOffsetX=-0.00137535, viewOffsetY=0.00017435)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.491548, 
    farPlane=0.516288, width=0.0109722, height=0.0419986, 
    viewOffsetX=-0.00199455, viewOffsetY=0.000429158)
session.viewports['Viewport: 2'].view.setValues(nearPlane=0.491085, 
    farPlane=0.515842, width=0.0100535, height=0.0419592, 
    viewOffsetX=0.000126873, viewOffsetY=0.000245205)
session.viewports['Viewport: 3'].view.setValues(nearPlane=0.492591, 
    farPlane=0.517337, width=0.0101915, height=0.0420824, 
    viewOffsetX=-0.000199634, viewOffsetY=0.000613504)
session.viewports['Viewport: 4'].view.setValues(nearPlane=0.492984, 
    farPlane=0.517754, width=0.00996965, height=0.0421195, 
    viewOffsetX=0.000289809, viewOffsetY=0.000797053)
session.viewports['Viewport: 10'].view.setValues(nearPlane=0.492093, 
    farPlane=0.516861, width=0.0108418, height=0.0420455, 
    viewOffsetX=-0.00167006, viewOffsetY=0.00042964)
session.viewports['Viewport: 11'].view.setValues(nearPlane=0.491348, 
    farPlane=0.515995, width=0.00993703, height=0.0419817, 
    viewOffsetX=0.000288865, viewOffsetY=0.000794448)
session.viewports['Viewport: 12'].view.setValues(nearPlane=0.491915, 
    farPlane=0.516711, width=0.0104981, height=0.0420267, 
    viewOffsetX=-0.000854316, viewOffsetY=0.000245602)
session.viewports['Viewport: 1'].view.setValues(width=0.0103138, 
    height=0.0394788, viewOffsetX=-0.00203302, viewOffsetY=0.000401442)
session.viewports['Viewport: 2'].view.setValues(width=0.00945028, 
    height=0.0394415, viewOffsetX=0.000116177, viewOffsetY=0.000215335)
session.viewports['Viewport: 3'].view.setValues(width=0.00957999, 
    height=0.0395573, viewOffsetX=-0.0002146, viewOffsetY=0.000587911)
session.viewports['Viewport: 4'].view.setValues(width=0.00937144, 
    height=0.0395922, viewOffsetX=0.000281263, viewOffsetY=0.000773611)
session.viewports['Viewport: 10'].view.setValues(width=0.0101913, 
    height=0.0395228, viewOffsetX=-0.00170429, viewOffsetY=0.000401893)
session.viewports['Viewport: 11'].view.setValues(width=0.00934078, 
    height=0.0394627, viewOffsetX=0.000280346, viewOffsetY=0.000771083)
session.viewports['Viewport: 12'].view.setValues(width=0.00986826, 
    height=0.039505, viewOffsetX=-0.00087788, viewOffsetY=0.000215684)
session.viewports['Viewport: 1'].view.setValues(width=0.00970935, 
    height=0.0371649, viewOffsetX=-0.00207225, viewOffsetY=0.000375945)
session.viewports['Viewport: 2'].view.setValues(width=0.00889643, 
    height=0.03713, viewOffsetX=0.000106281, viewOffsetY=0.000187535)
session.viewports['Viewport: 3'].view.setValues(width=0.00901849, 
    height=0.0372388, viewOffsetX=-0.000229006, viewOffsetY=0.000564689)
session.viewports['Viewport: 4'].view.setValues(width=0.00882215, 
    height=0.0372715, viewOffsetX=0.000273633, viewOffsetY=0.000752686)
session.viewports['Viewport: 10'].view.setValues(width=0.00959396, 
    height=0.0372064, viewOffsetX=-0.00173904, viewOffsetY=0.000376368)
session.viewports['Viewport: 11'].view.setValues(width=0.00879328, 
    height=0.0371496, viewOffsetX=0.000272742, viewOffsetY=0.000750226)
session.viewports['Viewport: 12'].view.setValues(width=0.00928991, 
    height=0.0371897, viewOffsetX=-0.000901367, viewOffsetY=0.00018784)
session.viewports['Viewport: 1'].view.setValues(width=0.00913917, 
    height=0.0349824, viewOffsetX=-0.00210915, viewOffsetY=0.000355675)
session.viewports['Viewport: 2'].view.setValues(width=0.00837401, 
    height=0.0349496, viewOffsetX=9.69486e-05, viewOffsetY=0.000165104)
session.viewports['Viewport: 3'].view.setValues(width=0.00848886, 
    height=0.0350519, viewOffsetX=-0.000242578, viewOffsetY=0.000546557)
session.viewports['Viewport: 4'].view.setValues(width=0.00830403, 
    height=0.0350826, viewOffsetX=0.000266431, viewOffsetY=0.000736711)
session.viewports['Viewport: 10'].view.setValues(width=0.00903055, 
    height=0.0350215, viewOffsetX=-0.00177174, viewOffsetY=0.000356075)
session.viewports['Viewport: 11'].view.setValues(width=0.00827686, 
    height=0.0349678, viewOffsetX=0.000265563, viewOffsetY=0.000734303)
session.viewports['Viewport: 12'].view.setValues(width=0.00874438, 
    height=0.0350058, viewOffsetX=-0.000923475, viewOffsetY=0.000165372)
session.viewports['Viewport: 1'].view.setValues(width=0.0086018, 
    height=0.0329255, viewOffsetX=-0.0021261, viewOffsetY=0.000350795)
session.viewports['Viewport: 2'].view.setValues(width=0.00788164, 
    height=0.0328947, viewOffsetX=0.000105993, viewOffsetY=0.000158198)
session.viewports['Viewport: 3'].view.setValues(width=0.0079897, 
    height=0.0329908, viewOffsetX=-0.000237538, viewOffsetY=0.000543697)
session.viewports['Viewport: 4'].view.setValues(width=0.00781573, 
    height=0.0330197, viewOffsetX=0.00027746, viewOffsetY=0.000735872)
session.viewports['Viewport: 10'].view.setValues(width=0.00849957, 
    height=0.0329623, viewOffsetX=-0.00178471, viewOffsetY=0.00035119)
session.viewports['Viewport: 11'].view.setValues(width=0.00779016, 
    height=0.0329116, viewOffsetX=0.000276557, viewOffsetY=0.000733467)
session.viewports['Viewport: 12'].view.setValues(width=0.00823024, 
    height=0.0329476, viewOffsetX=-0.000926444, viewOffsetY=0.000158455)
session.viewports['Viewport: 1'].view.setValues(width=0.00809542, 
    height=0.0309872, viewOffsetX=-0.00213873, viewOffsetY=0.000346197)
session.viewports['Viewport: 2'].view.setValues(width=0.00741767, 
    height=0.0309583, viewOffsetX=0.000117873, viewOffsetY=0.00015169)
session.viewports['Viewport: 3'].view.setValues(width=0.00751934, 
    height=0.0310486, viewOffsetX=-0.000229432, viewOffsetY=0.000541002)
session.viewports['Viewport: 4'].view.setValues(width=0.00735559, 
    height=0.0310757, viewOffsetX=0.000291207, viewOffsetY=0.000735081)
session.viewports['Viewport: 10'].view.setValues(width=0.0079992, 
    height=0.0310218, viewOffsetX=-0.00179358, viewOffsetY=0.000346586)
session.viewports['Viewport: 11'].view.setValues(width=0.00733153, 
    height=0.030974, viewOffsetX=0.000290259, viewOffsetY=0.000732679)
session.viewports['Viewport: 12'].view.setValues(width=0.00774574, 
    height=0.031008, viewOffsetX=-0.000925878, viewOffsetY=0.000151936)
session.viewports['Viewport: 1'].view.setValues(width=0.00761831, 
    height=0.0291609, viewOffsetX=-0.00206852, viewOffsetY=0.000285174)
session.viewports['Viewport: 2'].view.setValues(width=0.00698051, 
    height=0.0291338, viewOffsetX=0.000211228, viewOffsetY=8.88242e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.00707617, 
    height=0.0292186, viewOffsetX=-0.000139673, viewOffsetY=0.000481756)
session.viewports['Viewport: 4'].view.setValues(width=0.00692206, 
    height=0.0292441, viewOffsetX=0.000386214, viewOffsetY=0.000677677)
session.viewports['Viewport: 10'].view.setValues(width=0.00752776, 
    height=0.0291935, viewOffsetX=-0.00171974, viewOffsetY=0.000285495)
session.viewports['Viewport: 11'].view.setValues(width=0.00689941, 
    height=0.0291484, viewOffsetX=0.000384955, viewOffsetY=0.000675462)
session.viewports['Viewport: 12'].view.setValues(width=0.00728925, 
    height=0.0291806, viewOffsetX=-0.000843053, viewOffsetY=8.89698e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.00716884, 
    height=0.0274405, viewOffsetX=-0.00200239, viewOffsetY=0.000227687)
session.viewports['Viewport: 2'].view.setValues(width=0.00656869, 
    height=0.027415, viewOffsetX=0.000299174, viewOffsetY=2.96006e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.00665867, 
    height=0.0274947, viewOffsetX=-5.51147e-05, viewOffsetY=0.000425943)
session.viewports['Viewport: 4'].view.setValues(width=0.00651364, 
    height=0.0275186, viewOffsetX=0.000475717, viewOffsetY=0.000623598)
session.viewports['Viewport: 10'].view.setValues(width=0.00708363, 
    height=0.0274711, viewOffsetX=-0.00165018, viewOffsetY=0.000227943)
session.viewports['Viewport: 11'].view.setValues(width=0.00649233, 
    height=0.0274286, viewOffsetX=0.000474165, viewOffsetY=0.00062156)
session.viewports['Viewport: 12'].view.setValues(width=0.00685921, 
    height=0.027459, viewOffsetX=-0.000765025, viewOffsetY=2.9651e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.00674546, 
    height=0.0258199, viewOffsetX=-0.00194009, viewOffsetY=0.000173537)
session.viewports['Viewport: 2'].view.setValues(width=0.00618077, 
    height=0.0257959, viewOffsetX=0.000382016, viewOffsetY=-2.61849e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.00626541, 
    height=0.0258709, viewOffsetX=2.45344e-05, viewOffsetY=0.000373371)
session.viewports['Viewport: 4'].view.setValues(width=0.00612894, 
    height=0.0258934, viewOffsetX=0.000560023, viewOffsetY=0.000572659)
session.viewports['Viewport: 10'].view.setValues(width=0.00666529, 
    height=0.0258487, viewOffsetX=-0.00158466, viewOffsetY=0.000173733)
session.viewports['Viewport: 11'].view.setValues(width=0.00610889, 
    height=0.0258087, viewOffsetX=0.000558195, viewOffsetY=0.000570788)
session.viewports['Viewport: 12'].view.setValues(width=0.00645413, 
    height=0.0258374, viewOffsetX=-0.000691527, viewOffsetY=-2.62241e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.00634672, 
    height=0.0242936, viewOffsetX=-0.00188142, viewOffsetY=0.000122537)
session.viewports['Viewport: 2'].view.setValues(width=0.00581541, 
    height=0.0242711, viewOffsetX=0.000460038, viewOffsetY=-7.87255e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.00589503, 
    height=0.0243416, viewOffsetX=9.95502e-05, viewOffsetY=0.000323856)
session.viewports['Viewport: 4'].view.setValues(width=0.00576662, 
    height=0.0243627, viewOffsetX=0.000639424, viewOffsetY=0.000524684)
session.viewports['Viewport: 10'].view.setValues(width=0.00627129, 
    height=0.0243207, viewOffsetX=-0.00152295, viewOffsetY=0.000122676)
session.viewports['Viewport: 11'].view.setValues(width=0.00574776, 
    height=0.0242829, viewOffsetX=0.000637336, viewOffsetY=0.00052297)
session.viewports['Viewport: 12'].view.setValues(width=0.00607262, 
    height=0.0243101, viewOffsetX=-0.000622305, viewOffsetY=-7.88491e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.00597121, 
    height=0.0228563, viewOffsetX=-0.00182616, viewOffsetY=7.45089e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.00547135, 
    height=0.0228351, viewOffsetX=0.000533514, viewOffsetY=-0.000128204)
session.viewports['Viewport: 3'].view.setValues(width=0.00554624, 
    height=0.0229013, viewOffsetX=0.000170194, viewOffsetY=0.000277228)
session.viewports['Viewport: 4'].view.setValues(width=0.00542542, 
    height=0.0229212, viewOffsetX=0.000714197, viewOffsetY=0.000479504)
session.viewports['Viewport: 10'].view.setValues(width=0.00590024, 
    height=0.0228818, viewOffsetX=-0.00146484, viewOffsetY=7.45945e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.00540767, 
    height=0.0228462, viewOffsetX=0.000711865, viewOffsetY=0.000477939)
session.viewports['Viewport: 12'].view.setValues(width=0.00571334, 
    height=0.0228718, viewOffsetX=-0.000557116, viewOffsetY=-0.000128407)
session.viewports['Viewport: 1'].view.setValues(width=0.00561763, 
    height=0.0215029, viewOffsetX=-0.00177414, viewOffsetY=2.46402e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.00514737, 
    height=0.021483, viewOffsetX=0.000602701, viewOffsetY=-0.000179443)
session.viewports['Viewport: 3'].view.setValues(width=0.00521781, 
    height=0.0215452, viewOffsetX=0.000236713, viewOffsetY=0.000228676)
session.viewports['Viewport: 4'].view.setValues(width=0.00510414, 
    height=0.0215638, viewOffsetX=0.000784605, viewOffsetY=0.000432321)
session.viewports['Viewport: 10'].view.setValues(width=0.00555086, 
    height=0.0215268, viewOffsetX=-0.00141012, viewOffsetY=2.467e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.00508744, 
    height=0.0214933, viewOffsetX=0.000782043, viewOffsetY=0.00043091)
session.viewports['Viewport: 12'].view.setValues(width=0.00537503, 
    height=0.0215175, viewOffsetX=-0.000495733, viewOffsetY=-0.000179729)
session.viewports['Viewport: 1'].view.setValues(width=0.00528472, 
    height=0.0202286, viewOffsetX=-0.00176239, viewOffsetY=-1.35736e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.00484233, 
    height=0.0202099, viewOffsetX=0.000630576, viewOffsetY=-0.00021894)
session.viewports['Viewport: 3'].view.setValues(width=0.00490859, 
    height=0.0202684, viewOffsetX=0.000262096, viewOffsetY=0.000191704)
session.viewports['Viewport: 4'].view.setValues(width=0.00480165, 
    height=0.0202859, viewOffsetX=0.000813679, viewOffsetY=0.000396631)
session.viewports['Viewport: 10'].view.setValues(width=0.00522191, 
    height=0.0202511, viewOffsetX=-0.00139588, viewOffsetY=-1.35863e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.00478594, 
    height=0.0202195, viewOffsetX=0.000811022, viewOffsetY=0.000395336)
session.viewports['Viewport: 12'].view.setValues(width=0.0050565, 
    height=0.0202424, viewOffsetX=-0.000475265, viewOffsetY=-0.000219289)
session.viewports['Viewport: 1'].view.setValues(width=0.00497131, 
    height=0.0190289, viewOffsetX=-0.00175133, viewOffsetY=-4.95492e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.00455517, 
    height=0.0190114, viewOffsetX=0.00065682, viewOffsetY=-0.000256124)
session.viewports['Viewport: 3'].view.setValues(width=0.00461748, 
    height=0.0190663, viewOffsetX=0.000285991, viewOffsetY=0.000156898)
session.viewports['Viewport: 4'].view.setValues(width=0.00451688, 
    height=0.0190828, viewOffsetX=0.000841051, viewOffsetY=0.000363031)
session.viewports['Viewport: 10'].view.setValues(width=0.00491222, 
    height=0.0190501, viewOffsetX=-0.00138247, viewOffsetY=-4.96022e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.0045021, 
    height=0.0190203, viewOffsetX=0.000838304, viewOffsetY=0.000361847)
session.viewports['Viewport: 12'].view.setValues(width=0.00475663, 
    height=0.0190419, viewOffsetX=-0.000455995, viewOffsetY=-0.000256533)
session.viewports['Viewport: 1'].view.setValues(width=0.00467628, 
    height=0.0178996, viewOffsetX=-0.00174092, viewOffsetY=-8.3415e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.00428484, 
    height=0.0178831, viewOffsetX=0.000681524, viewOffsetY=-0.000291127)
session.viewports['Viewport: 3'].view.setValues(width=0.00434345, 
    height=0.0179348, viewOffsetX=0.000308486, viewOffsetY=0.000124133)
session.viewports['Viewport: 4'].view.setValues(width=0.00424881, 
    height=0.0179502, viewOffsetX=0.000866816, viewOffsetY=0.000331402)
session.viewports['Viewport: 10'].view.setValues(width=0.0046207, 
    height=0.0179196, viewOffsetX=-0.00136985, viewOffsetY=-8.35058e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.00423491, 
    height=0.0178915, viewOffsetX=0.000863985, viewOffsetY=0.000330322)
session.viewports['Viewport: 12'].view.setValues(width=0.00447435, 
    height=0.0179119, viewOffsetX=-0.000437856, viewOffsetY=-0.000291592)
session.viewports['Viewport: 1'].view.setValues(width=0.00439858, 
    height=0.0168366, viewOffsetX=-0.00173112, viewOffsetY=-0.000115292)
session.viewports['Viewport: 2'].view.setValues(width=0.00403039, 
    height=0.0168212, viewOffsetX=0.000704777, viewOffsetY=-0.000324075)
session.viewports['Viewport: 3'].view.setValues(width=0.00408551, 
    height=0.0168697, viewOffsetX=0.000329659, viewOffsetY=9.32928e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.00399648, 
    height=0.0168842, viewOffsetX=0.000891069, viewOffsetY=0.000301631)
session.viewports['Viewport: 10'].view.setValues(width=0.0043463, 
    height=0.0168554, viewOffsetX=-0.00135797, viewOffsetY=-0.000115418)
session.viewports['Viewport: 11'].view.setValues(width=0.00398341, 
    height=0.016829, viewOffsetX=0.000888159, viewOffsetY=0.000300648)
session.viewports['Viewport: 12'].view.setValues(width=0.00420864, 
    height=0.0168482, viewOffsetX=-0.000420782, viewOffsetY=-0.000324592)
session.viewports['Viewport: 1'].view.setValues(width=0.00413721, 
    height=0.0158362, viewOffsetX=-0.00172189, viewOffsetY=-0.000145294)
session.viewports['Viewport: 2'].view.setValues(width=0.0037909, 
    height=0.0158216, viewOffsetX=0.000726663, viewOffsetY=-0.000355085)
session.viewports['Viewport: 3'].view.setValues(width=0.00384273, 
    height=0.0158673, viewOffsetX=0.000349586, viewOffsetY=6.42661e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.003759, 
    height=0.0158809, viewOffsetX=0.000913895, viewOffsetY=0.000273611)
session.viewports['Viewport: 10'].view.setValues(width=0.00408804, 
    height=0.0158539, viewOffsetX=-0.00134679, viewOffsetY=-0.000145454)
session.viewports['Viewport: 11'].view.setValues(width=0.0037467, 
    height=0.015829, viewOffsetX=0.00091091, viewOffsetY=0.000272719)
session.viewports['Viewport: 12'].view.setValues(width=0.00395856, 
    height=0.0158471, viewOffsetX=-0.000404712, viewOffsetY=-0.000355652)
session.viewports['Viewport: 1'].view.setValues(width=0.00389123, 
    height=0.0148946, viewOffsetX=-0.00171321, viewOffsetY=-0.00017353)
session.viewports['Viewport: 2'].view.setValues(width=0.00356551, 
    height=0.014881, viewOffsetX=0.00074726, viewOffsetY=-0.000384269)
session.viewports['Viewport: 3'].view.setValues(width=0.00361426, 
    height=0.0149239, viewOffsetX=0.000368341, viewOffsetY=3.69485e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.0035355, 
    height=0.0149367, viewOffsetX=0.000935378, viewOffsetY=0.000247241)
session.viewports['Viewport: 10'].view.setValues(width=0.00384498, 
    height=0.0149113, viewOffsetX=-0.00133626, viewOffsetY=-0.000173721)
session.viewports['Viewport: 11'].view.setValues(width=0.00352393, 
    height=0.0148878, viewOffsetX=0.000932322, viewOffsetY=0.000246435)
session.viewports['Viewport: 12'].view.setValues(width=0.00372321, 
    height=0.0149049, viewOffsetX=-0.000389589, viewOffsetY=-0.000384883)
session.viewports['Viewport: 1'].view.setValues(width=0.00365975, 
    height=0.0140086, viewOffsetX=-0.00171566, viewOffsetY=-0.000128992)
session.viewports['Viewport: 2'].view.setValues(width=0.00335341, 
    height=0.0139957, viewOffsetX=0.000756017, viewOffsetY=-0.000340569)
session.viewports['Viewport: 3'].view.setValues(width=0.00339925, 
    height=0.0140361, viewOffsetX=0.000375369, viewOffsetY=8.237e-05)
session.viewports['Viewport: 4'].view.setValues(width=0.00332517, 
    height=0.0140481, viewOffsetX=0.000944981, viewOffsetY=0.000293495)
session.viewports['Viewport: 10'].view.setValues(width=0.00361625, 
    height=0.0140242, viewOffsetY=-0.000129133)
session.viewports['Viewport: 11'].view.setValues(width=0.00331429, 
    height=0.0140021, viewOffsetX=0.000941895, viewOffsetY=0.000292538)
session.viewports['Viewport: 12'].view.setValues(width=0.00350172, 
    height=0.0140182, viewOffsetX=-0.000386, viewOffsetY=-0.000341112)
session.viewports['Viewport: 1'].view.setValues(width=0.00344192, 
    height=0.0131748, viewOffsetX=-0.00171797, viewOffsetY=-8.56587e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.00315382, 
    height=0.0131627, viewOffsetX=0.000764257, viewOffsetY=-0.000298022)
session.viewports['Viewport: 3'].view.setValues(width=0.00319693, 
    height=0.0132006, viewOffsetX=0.000381982, viewOffsetY=0.000126534)
session.viewports['Viewport: 4'].view.setValues(width=0.00312726, 
    height=0.0132119, viewOffsetX=0.000954018, viewOffsetY=0.000338441)
session.viewports['Viewport: 10'].view.setValues(width=0.00340101, 
    height=0.0131895, viewOffsetY=-8.57519e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.00311703, 
    height=0.0131687, viewOffsetX=0.000950902, viewOffsetY=0.000337337)
session.viewports['Viewport: 12'].view.setValues(width=0.00329331, 
    height=0.0131839, viewOffsetX=-0.000382623, viewOffsetY=-0.000298498)
session.viewports['Viewport: 1'].view.setValues(width=0.00323697, 
    height=0.0123903, viewOffsetX=-0.00172014, viewOffsetY=-4.48853e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.00296602, 
    height=0.0123789, viewOffsetX=0.000772011, viewOffsetY=-0.000257989)
session.viewports['Viewport: 3'].view.setValues(width=0.00300656, 
    height=0.0124146, viewOffsetX=0.000388205, viewOffsetY=0.00016809)
session.viewports['Viewport: 4'].view.setValues(width=0.00294103, 
    height=0.0124252, viewOffsetX=0.000962522, viewOffsetY=0.000380733)
session.viewports['Viewport: 10'].view.setValues(width=0.00319849, 
    height=0.0124041, viewOffsetY=-4.4933e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.00293141, 
    height=0.0123845, viewOffsetX=0.000959378, viewOffsetY=0.000379491)
session.viewports['Viewport: 12'].view.setValues(width=0.0030972, 
    height=0.0123988, viewOffsetX=-0.000379445, viewOffsetY=-0.000258401)
session.viewports['Viewport: 1'].view.setValues(width=0.00304413, 
    height=0.0116521, viewOffsetX=-0.00171713, viewOffsetY=-6.52246e-06)
session.viewports['Viewport: 2'].view.setValues(width=0.00278933, 
    height=0.0116415, viewOffsetX=0.000784357, viewOffsetY=-0.000220323)
session.viewports['Viewport: 3'].view.setValues(width=0.00282744, 
    height=0.011675, viewOffsetX=0.000399108, viewOffsetY=0.000207189)
session.viewports['Viewport: 4'].view.setValues(width=0.00276582, 
    height=0.0116849, viewOffsetX=0.000975566, viewOffsetY=0.000420524)
session.viewports['Viewport: 10'].view.setValues(width=0.00300794, 
    height=0.0116651, viewOffsetX=-0.00133387, viewOffsetY=-6.52741e-06)
session.viewports['Viewport: 11'].view.setValues(width=0.00275677, 
    height=0.0116467, viewOffsetX=0.00097238, viewOffsetY=0.000419151)
session.viewports['Viewport: 12'].view.setValues(width=0.00291269, 
    height=0.0116602, viewOffsetX=-0.000371396, viewOffsetY=-0.000220674)
session.viewports['Viewport: 1'].view.setValues(width=0.0028627, 
    height=0.0109577, viewOffsetX=-0.00170006, viewOffsetY=3.19373e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.00262309, 
    height=0.0109477, viewOffsetX=0.000810222, viewOffsetY=-0.000182517)
session.viewports['Viewport: 3'].view.setValues(width=0.00265892, 
    height=0.0109791, viewOffsetX=0.000423608, viewOffsetY=0.000246342)
session.viewports['Viewport: 4'].view.setValues(width=0.00260097, 
    height=0.0109885, viewOffsetX=0.00100207, viewOffsetY=0.000460326)
session.viewports['Viewport: 10'].view.setValues(width=0.00282867, 
    height=0.0109699, viewOffsetX=-0.00131543, viewOffsetY=3.19752e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.00259246, 
    height=0.0109526, viewOffsetX=0.000998796, viewOffsetY=0.000458823)
session.viewports['Viewport: 12'].view.setValues(width=0.0027391, 
    height=0.0109653, viewOffsetX=-0.000349552, viewOffsetY=-0.000182807)
session.viewports['Viewport: 1'].view.setValues(width=0.00269201, 
    height=0.0103043, viewOffsetX=-0.00165611, viewOffsetY=7.25709e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.00246669, 
    height=0.0102949, viewOffsetX=0.000862471, viewOffsetY=-0.000142494)
session.viewports['Viewport: 3'].view.setValues(width=0.00250039, 
    height=0.0103245, viewOffsetX=0.000474559, viewOffsetY=0.000287628)
session.viewports['Viewport: 4'].view.setValues(width=0.00244589, 
    height=0.0103333, viewOffsetX=0.00105488, viewOffsetY=0.000502219)
session.viewports['Viewport: 10'].view.setValues(width=0.00266002, 
    height=0.0103158, viewOffsetX=-0.00127016, viewOffsetY=7.26542e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.00243789, 
    height=0.0102995, viewOffsetX=0.00105144, viewOffsetY=0.00050058)
session.viewports['Viewport: 12'].view.setValues(width=0.00257579, 
    height=0.0103115, viewOffsetX=-0.00030104, viewOffsetY=-0.00014272)
session.viewports['Viewport: 1'].view.setValues(width=0.00286492, 
    height=0.0109662, viewOffsetX=-0.00167348, viewOffsetY=8.98547e-05)
session.viewports['Viewport: 2'].view.setValues(width=0.00262513, 
    height=0.0109562, viewOffsetX=0.00083878, viewOffsetY=-0.000124721)
session.viewports['Viewport: 3'].view.setValues(width=0.00266099, 
    height=0.0109877, viewOffsetX=0.000451849, viewOffsetY=0.000304441)
session.viewports['Viewport: 4'].view.setValues(width=0.00260299, 
    height=0.010997, viewOffsetX=0.00103073, viewOffsetY=0.000518542)
session.viewports['Viewport: 10'].view.setValues(width=0.00283087, 
    height=0.0109784, viewOffsetX=-0.00128852, viewOffsetY=8.99573e-05)
session.viewports['Viewport: 11'].view.setValues(width=0.00259447, 
    height=0.010961, viewOffsetX=0.00102737, viewOffsetY=0.000516849)
session.viewports['Viewport: 12'].view.setValues(width=0.00274123, 
    height=0.0109738, viewOffsetX=-0.000321852, viewOffsetY=-0.000124918)
session.viewports['Viewport: 1'].view.setValues(width=0.00304663, 
    height=0.0116617, viewOffsetX=-0.00169065, viewOffsetY=0.000108171)
session.viewports['Viewport: 2'].view.setValues(width=0.00279162, 
    height=0.0116511, viewOffsetX=0.000812919, viewOffsetY=-0.000105716)
session.viewports['Viewport: 3'].view.setValues(width=0.00282976, 
    height=0.0116845, viewOffsetX=0.000427337, viewOffsetY=0.000322089)
session.viewports['Viewport: 4'].view.setValues(width=0.00276808, 
    height=0.0116945, viewOffsetX=0.00100424, viewOffsetY=0.000535502)
session.viewports['Viewport: 10'].view.setValues(width=0.00301042, 
    height=0.0116747, viewOffsetX=-0.00130704, viewOffsetY=0.000108294)
session.viewports['Viewport: 11'].view.setValues(width=0.00275903, 
    height=0.0116563, viewOffsetX=0.00100096, viewOffsetY=0.000533754)
session.viewports['Viewport: 12'].view.setValues(width=0.00291509, 
    height=0.0116698, viewOffsetX=-0.000343741, viewOffsetY=-0.000105882)
session.viewports['Viewport: 1'].view.setValues(width=0.00346069, 
    height=0.0132466, viewOffsetX=-0.00173042, viewOffsetY=0.000149818)
session.viewports['Viewport: 2'].view.setValues(width=0.00317102, 
    height=0.0132345, viewOffsetX=0.000754569, viewOffsetY=-6.26048e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.00321435, 
    height=0.0132726, viewOffsetX=0.000371871, viewOffsetY=0.000362315)
session.viewports['Viewport: 4'].view.setValues(width=0.00314429, 
    height=0.0132839, viewOffsetX=0.00094455, viewOffsetY=0.000574266)
session.viewports['Viewport: 10'].view.setValues(width=0.00341956, 
    height=0.0132614, viewOffsetX=-0.00134971, viewOffsetY=0.000149987)
session.viewports['Viewport: 11'].view.setValues(width=0.00313401, 
    height=0.0132405, viewOffsetX=0.000941465, viewOffsetY=0.000572391)
session.viewports['Viewport: 12'].view.setValues(width=0.00331127, 
    height=0.0132558, viewOffsetX=-0.00039361, viewOffsetY=-6.27023e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.00367822, 
    height=0.0140793, viewOffsetX=-0.00175033, viewOffsetY=0.000171836)
session.viewports['Viewport: 2'].view.setValues(width=0.00337034, 
    height=0.0140664, viewOffsetX=0.000723043, viewOffsetY=-3.96598e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.0034164, 
    height=0.0141069, viewOffsetX=0.000342145, viewOffsetY=0.000383431)
session.viewports['Viewport: 4'].view.setValues(width=0.00334194, 
    height=0.0141189, viewOffsetX=0.00091218, viewOffsetY=0.000594457)
session.viewports['Viewport: 10'].view.setValues(width=0.0036345, 
    height=0.014095, viewOffsetX=-0.00137143, viewOffsetY=0.00017203)
session.viewports['Viewport: 11'].view.setValues(width=0.00333101, 
    height=0.0140728, viewOffsetX=0.000909201, viewOffsetY=0.000592516)
session.viewports['Viewport: 12'].view.setValues(width=0.0035194, 
    height=0.014089, viewOffsetX=-0.000419825, viewOffsetY=-3.97205e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.00391112, 
    height=0.0149708, viewOffsetX=-0.0017434, viewOffsetY=0.000196832)
session.viewports['Viewport: 2'].view.setValues(width=0.00358374, 
    height=0.0149571, viewOffsetX=0.000718882, viewOffsetY=-1.37827e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.00363272, 
    height=0.0150001, viewOffsetX=0.000339695, viewOffsetY=0.000407571)
session.viewports['Viewport: 4'].view.setValues(width=0.00355355, 
    height=0.0150129, viewOffsetX=0.000907174, viewOffsetY=0.000617717)
session.viewports['Viewport: 10'].view.setValues(width=0.00386463, 
    height=0.0149875, viewOffsetX=-0.0013662, viewOffsetY=0.000197054)
session.viewports['Viewport: 11'].view.setValues(width=0.00354193, 
    height=0.0149638, viewOffsetX=0.000904211, viewOffsetY=0.0006157)
session.viewports['Viewport: 12'].view.setValues(width=0.00374224, 
    height=0.0149811, viewOffsetX=-0.000418859, viewOffsetY=-1.38019e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.00415862, 
    height=0.0159181, viewOffsetX=-0.00171171, viewOffsetY=0.000226627)
session.viewports['Viewport: 2'].view.setValues(width=0.00381052, 
    height=0.0159035, viewOffsetX=0.00073878, viewOffsetY=1.69533e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.00386261, 
    height=0.0159494, viewOffsetX=0.000361399, viewOffsetY=0.000436459)
session.viewports['Viewport: 4'].view.setValues(width=0.00377844, 
    height=0.015963, viewOffsetX=0.00092614, viewOffsetY=0.000645665)
session.viewports['Viewport: 10'].view.setValues(width=0.00410919, 
    height=0.0159359, viewOffsetX=-0.0013363, viewOffsetY=0.000226882)
session.viewports['Viewport: 11'].view.setValues(width=0.00376608, 
    height=0.0159108, viewOffsetX=0.000923115, viewOffsetY=0.000643556)
session.viewports['Viewport: 12'].view.setValues(width=0.00397906, 
    height=0.0159291, viewOffsetX=-0.000393469, viewOffsetY=1.69835e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.00442164, 
    height=0.0169249, viewOffsetX=-0.00167804, viewOffsetY=0.000258291)
session.viewports['Viewport: 2'].view.setValues(width=0.00405153, 
    height=0.0169094, viewOffsetX=0.000759926, viewOffsetY=4.96164e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.00410692, 
    height=0.0169581, viewOffsetX=0.000384465, viewOffsetY=0.000467157)
session.viewports['Viewport: 4'].view.setValues(width=0.00401742, 
    height=0.0169727, viewOffsetX=0.000946296, viewOffsetY=0.000675365)
session.viewports['Viewport: 10'].view.setValues(width=0.00436909, 
    height=0.0169438, viewOffsetX=-0.00130452, viewOffsetY=0.000258582)
session.viewports['Viewport: 11'].view.setValues(width=0.00400428, 
    height=0.0169172, viewOffsetX=0.000943205, viewOffsetY=0.000673159)
session.viewports['Viewport: 12'].view.setValues(width=0.00423072, 
    height=0.0169365, viewOffsetX=-0.000366488, viewOffsetY=4.96991e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.00470114, 
    height=0.0179948, viewOffsetX=-0.00164227, viewOffsetY=0.000291938)
session.viewports['Viewport: 2'].view.setValues(width=0.00430762, 
    height=0.0179782, viewOffsetX=0.000782396, viewOffsetY=8.43251e-05)
session.viewports['Viewport: 3'].view.setValues(width=0.00436652, 
    height=0.0180301, viewOffsetX=0.000408975, viewOffsetY=0.000499779)
session.viewports['Viewport: 4'].view.setValues(width=0.00427137, 
    height=0.0180456, viewOffsetX=0.000967714, viewOffsetY=0.000706926)
session.viewports['Viewport: 10'].view.setValues(width=0.00464526, 
    height=0.0180148, viewOffsetX=-0.00127076, viewOffsetY=0.000292266)
session.viewports['Viewport: 11'].view.setValues(width=0.0042574, 
    height=0.0179865, viewOffsetX=0.000964553, viewOffsetY=0.000704617)
session.viewports['Viewport: 12'].view.setValues(width=0.00449814, 
    height=0.0180071, viewOffsetX=-0.000337816, viewOffsetY=8.44636e-05)
session.viewports['Viewport: 1'].view.setValues(width=0.00499811, 
    height=0.0191315, viewOffsetX=-0.00160425, viewOffsetY=0.00032769)
session.viewports['Viewport: 2'].view.setValues(width=0.00457973, 
    height=0.0191139, viewOffsetX=0.000806271, viewOffsetY=0.000121205)
session.viewports['Viewport: 3'].view.setValues(width=0.00464237, 
    height=0.0191691, viewOffsetX=0.000435018, viewOffsetY=0.00053444)
session.viewports['Viewport: 4'].view.setValues(width=0.00454121, 
    height=0.0191856, viewOffsetX=0.000990472, viewOffsetY=0.000740461)
session.viewports['Viewport: 10'].view.setValues(width=0.00493871, 
    height=0.0191528, viewOffsetX=-0.00123488, viewOffsetY=0.000328058)
session.viewports['Viewport: 11'].view.setValues(width=0.00452636, 
    height=0.0191228, viewOffsetX=0.000987236, viewOffsetY=0.000738042)
session.viewports['Viewport: 12'].view.setValues(width=0.00478229, 
    height=0.0191446, viewOffsetX=-0.000307351, viewOffsetY=0.000121402)
session.viewports['Viewport: 1'].view.setValues(width=0.00567594, 
    height=0.021726, viewOffsetX=-0.00151916, viewOffsetY=0.000409085)
session.viewports['Viewport: 2'].view.setValues(width=0.00520081, 
    height=0.021706, viewOffsetX=0.000861067, viewOffsetY=0.000205006)
session.viewports['Viewport: 3'].view.setValues(width=0.00527195, 
    height=0.0217687, viewOffsetX=0.000494456, viewOffsetY=0.000613516)
session.viewports['Viewport: 4'].view.setValues(width=0.00515708, 
    height=0.0217875, viewOffsetX=0.00104287, viewOffsetY=0.000817132)
session.viewports['Viewport: 10'].view.setValues(width=0.00560847, 
    height=0.0217503, viewOffsetX=-0.00115437, viewOffsetY=0.000409543)
session.viewports['Viewport: 11'].view.setValues(width=0.00514021, 
    height=0.0217162, viewOffsetX=0.00103946, viewOffsetY=0.000814462)
session.viewports['Viewport: 12'].view.setValues(width=0.00543083, 
    height=0.0217409, viewOffsetX=-0.000238432, viewOffsetY=0.000205338)
session.viewports['Viewport: 1'].view.setValues(width=0.00602915, 
    height=0.0230781, viewOffsetX=-0.00147225, viewOffsetY=0.000451814)
session.viewports['Viewport: 2'].view.setValues(width=0.00552445, 
    height=0.0230568, viewOffsetX=0.000889156, viewOffsetY=0.000249248)
session.viewports['Viewport: 3'].view.setValues(width=0.00560004, 
    height=0.0231235, viewOffsetX=0.000525435, viewOffsetY=0.00065478)
session.viewports['Viewport: 4'].view.setValues(width=0.00547804, 
    height=0.0231434, viewOffsetX=0.00106948, viewOffsetY=0.000856887)
session.viewports['Viewport: 10'].view.setValues(width=0.00595749, 
    height=0.0231038, viewOffsetX=-0.0011103, viewOffsetY=0.000452321)
session.viewports['Viewport: 11'].view.setValues(width=0.00546012, 
    height=0.0230677, viewOffsetX=0.00106598, viewOffsetY=0.000854087)
session.viewports['Viewport: 12'].view.setValues(width=0.00576879, 
    height=0.0230938, viewOffsetX=-0.000201577, viewOffsetY=0.000249651)
session.viewports['Viewport: 1'].view.setValues(width=0.00640896, 
    height=0.0245319, viewOffsetX=-0.00142365, viewOffsetY=0.000497535)
session.viewports['Viewport: 2'].view.setValues(width=0.00587246, 
    height=0.0245092, viewOffsetX=0.000919694, viewOffsetY=0.000296409)
session.viewports['Viewport: 3'].view.setValues(width=0.00595283, 
    height=0.0245802, viewOffsetX=0.000558742, viewOffsetY=0.00069911)
session.viewports['Viewport: 4'].view.setValues(width=0.00582314, 
    height=0.0246014, viewOffsetX=0.00109859, viewOffsetY=0.000899778)
session.viewports['Viewport: 10'].view.setValues(width=0.00633279, 
    height=0.0245592, viewOffsetX=-0.00106444, viewOffsetY=0.000498093)
session.viewports['Viewport: 11'].view.setValues(width=0.0058041, 
    height=0.024521, viewOffsetX=0.001095, viewOffsetY=0.000896838)
session.viewports['Viewport: 12'].view.setValues(width=0.00613218, 
    height=0.0245486, viewOffsetX=-0.000162623, viewOffsetY=0.000296888)
session.viewports['Viewport: 1'].view.setValues(width=0.00681229, 
    height=0.0260757, viewOffsetX=-0.00137202, viewOffsetY=0.00054609)
session.viewports['Viewport: 2'].view.setValues(width=0.00624201, 
    height=0.0260516, viewOffsetX=0.000952118, viewOffsetY=0.000346495)
session.viewports['Viewport: 3'].view.setValues(width=0.00632746, 
    height=0.0261271, viewOffsetX=0.000594112, viewOffsetY=0.000746185)
session.viewports['Viewport: 4'].view.setValues(width=0.00618962, 
    height=0.0261497, viewOffsetX=0.0011295, viewOffsetY=0.000945323)
session.viewports['Viewport: 10'].view.setValues(width=0.00673132, 
    height=0.0261048, viewOffsetX=-0.00101571, viewOffsetY=0.000546701)
session.viewports['Viewport: 11'].view.setValues(width=0.00616938, 
    height=0.0260642, viewOffsetX=0.00112581, viewOffsetY=0.000942234)
session.viewports['Viewport: 12'].view.setValues(width=0.00651809, 
    height=0.0260934, viewOffsetX=-0.000121249, viewOffsetY=0.000347055)
session.viewports['Viewport: 1'].view.setValues(width=0.00724061, 
    height=0.0277152, viewOffsetX=-0.00131719, viewOffsetY=0.000597654)
session.viewports['Viewport: 2'].view.setValues(width=0.00663447, 
    height=0.0276895, viewOffsetX=0.000986552, viewOffsetY=0.000399686)
session.viewports['Viewport: 3'].view.setValues(width=0.00672531, 
    height=0.0277699, viewOffsetX=0.000631675, viewOffsetY=0.000796178)
session.viewports['Viewport: 4'].view.setValues(width=0.00657882, 
    height=0.027794, viewOffsetX=0.00116232, viewOffsetY=0.000993691)
session.viewports['Viewport: 10'].view.setValues(width=0.00715455, 
    height=0.0277461, viewOffsetX=-0.000963963, viewOffsetY=0.000598323)
session.viewports['Viewport: 11'].view.setValues(width=0.0065573, 
    height=0.0277031, viewOffsetX=0.00115852, viewOffsetY=0.000990444)
session.viewports['Viewport: 12'].view.setValues(width=0.0069279, 
    height=0.027734, viewOffsetX=-7.73101e-05, viewOffsetY=0.00040033)
session.viewports['Viewport: 1'].view.setValues(width=0.00769543, 
    height=0.0294562, viewOffsetX=-0.00125897, viewOffsetY=0.000652408)
session.viewports['Viewport: 2'].view.setValues(width=0.0070512, 
    height=0.0294288, viewOffsetX=0.00102312, viewOffsetY=0.000456166)
session.viewports['Viewport: 3'].view.setValues(width=0.00714778, 
    height=0.0295143, viewOffsetX=0.000671561, viewOffsetY=0.000849264)
session.viewports['Viewport: 4'].view.setValues(width=0.00699209, 
    height=0.02954, viewOffsetX=0.00119718, viewOffsetY=0.00104505)
session.viewports['Viewport: 10'].view.setValues(width=0.00760397, 
    height=0.029489, viewOffsetX=-0.000909016, viewOffsetY=0.000653138)
session.viewports['Viewport: 11'].view.setValues(width=0.00696922, 
    height=0.0294433, viewOffsetX=0.00119326, viewOffsetY=0.00104164)
session.viewports['Viewport: 12'].view.setValues(width=0.00736307, 
    height=0.0294761, viewOffsetX=-3.06536e-05, viewOffsetY=0.000456902)
session.viewports['Viewport: 1'].view.setValues(width=0.00817833, 
    height=0.0313046, viewOffsetX=-0.00119715, viewOffsetY=0.000710542)
session.viewports['Viewport: 2'].view.setValues(width=0.00749366, 
    height=0.0312754, viewOffsetX=0.00106194, viewOffsetY=0.000516133)
session.viewports['Viewport: 3'].view.setValues(width=0.00759632, 
    height=0.0313665, viewOffsetX=0.000713909, viewOffsetY=0.000905627)
session.viewports['Viewport: 4'].view.setValues(width=0.00743088, 
    height=0.0313937, viewOffsetX=0.00123418, viewOffsetY=0.00109958)
session.viewports['Viewport: 10'].view.setValues(width=0.00808113, 
    height=0.0313395, viewOffsetX=-0.000850676, viewOffsetY=0.000711337)
session.viewports['Viewport: 11'].view.setValues(width=0.00740657, 
    height=0.031291, viewOffsetX=0.00123015, viewOffsetY=0.00109599)
session.viewports['Viewport: 12'].view.setValues(width=0.0078251, 
    height=0.0313257, viewOffsetX=1.88833e-05, viewOffsetY=0.000516966)
session.viewports['Viewport: 1'].view.setValues(width=0.00869098, 
    height=0.0332669, viewOffsetX=-0.00113152, viewOffsetY=0.000772258)
session.viewports['Viewport: 2'].view.setValues(width=0.00796338, 
    height=0.0332358, viewOffsetX=0.00110315, viewOffsetY=0.000579795)
session.viewports['Viewport: 3'].view.setValues(width=0.0080725, 
    height=0.0333327, viewOffsetX=0.000758867, viewOffsetY=0.000965463)
session.viewports['Viewport: 4'].view.setValues(width=0.0078967, 
    height=0.0333617, viewOffsetX=0.00127347, viewOffsetY=0.00115748)
session.viewports['Viewport: 10'].view.setValues(width=0.00858768, 
    height=0.033304, viewOffsetX=-0.000788742, viewOffsetY=0.000773121)
session.viewports['Viewport: 11'].view.setValues(width=0.00787087, 
    height=0.0332526, viewOffsetX=0.00126931, viewOffsetY=0.00115369)
session.viewports['Viewport: 12'].view.setValues(width=0.00831559, 
    height=0.0332893, viewOffsetX=7.14718e-05, viewOffsetY=0.00058073)
session.viewports['Viewport: 1'].view.setValues(width=0.00923514, 
    height=0.0353498, viewOffsetX=-0.00106186, viewOffsetY=0.000837767)
session.viewports['Viewport: 2'].view.setValues(width=0.00846197, 
    height=0.0353167, viewOffsetX=0.0011469, viewOffsetY=0.000647369)
session.viewports['Viewport: 3'].view.setValues(width=0.00857796, 
    height=0.0354198, viewOffsetX=0.000806588, viewOffsetY=0.00102898)
session.viewports['Viewport: 4'].view.setValues(width=0.00839116, 
    height=0.0354507, viewOffsetX=0.00131517, viewOffsetY=0.00121893)
session.viewports['Viewport: 10'].view.setValues(width=0.00912537, 
    height=0.0353892, viewOffsetX=-0.000723, viewOffsetY=0.000838703)
session.viewports['Viewport: 11'].view.setValues(width=0.00836371, 
    height=0.0353347, viewOffsetX=0.00131087, viewOffsetY=0.00121494)
session.viewports['Viewport: 12'].view.setValues(width=0.00883623, 
    height=0.0353735, viewOffsetX=0.000127293, viewOffsetY=0.000648413)
session.viewports['Viewport: 1'].view.setValues(width=0.00981266, 
    height=0.0375604, viewOffsetX=-0.000987934, viewOffsetY=0.000907292)
session.viewports['Viewport: 2'].view.setValues(width=0.00899112, 
    height=0.0375252, viewOffsetX=0.00119332, viewOffsetY=0.000719087)
session.viewports['Viewport: 3'].view.setValues(width=0.0091144, 
    height=0.0376349, viewOffsetX=0.000857236, viewOffsetY=0.00109639)
session.viewports['Viewport: 4'].view.setValues(width=0.00891594, 
    height=0.0376678, viewOffsetX=0.00135943, viewOffsetY=0.00128415)
session.viewports['Viewport: 10'].view.setValues(width=0.00969603, 
    height=0.0376023, viewOffsetX=-0.000653228, viewOffsetY=0.000908307)
session.viewports['Viewport: 11'].view.setValues(width=0.00888677, 
    height=0.0375446, viewOffsetX=0.00135498, viewOffsetY=0.00127995)
session.viewports['Viewport: 12'].view.setValues(width=0.00938879, 
    height=0.0375855, viewOffsetX=0.000186536, viewOffsetY=0.000720245)
session.viewports['Viewport: 1'].view.setValues(width=0.0104255, 
    height=0.0399062, viewOffsetX=-0.000909481, viewOffsetY=0.00098107)
session.viewports['Viewport: 2'].view.setValues(width=0.00955264, 
    height=0.0398687, viewOffsetX=0.00124259, viewOffsetY=0.00079519)
session.viewports['Viewport: 3'].view.setValues(width=0.00968366, 
    height=0.0399854, viewOffsetX=0.000910981, viewOffsetY=0.00116792)
session.viewports['Viewport: 4'].view.setValues(width=0.00947282, 
    height=0.0400205, viewOffsetX=0.00140639, viewOffsetY=0.00135335)
session.viewports['Viewport: 10'].view.setValues(width=0.0103016, 
    height=0.0399507, viewOffsetX=-0.000579188, viewOffsetY=0.000982167)
session.viewports['Viewport: 11'].view.setValues(width=0.00944183, 
    height=0.0398896, viewOffsetX=0.0014018, viewOffsetY=0.00134893)
session.viewports['Viewport: 12'].view.setValues(width=0.00997514, 
    height=0.0399328, viewOffsetX=0.000249402, viewOffsetY=0.000796471)
session.viewports['Viewport: 1'].view.setValues(width=0.0110757, 
    height=0.0423951, viewOffsetX=-0.000826242, viewOffsetY=0.00105935)
session.viewports['Viewport: 2'].view.setValues(width=0.0101484, 
    height=0.0423552, viewOffsetX=0.00129486, viewOffsetY=0.000875935)
session.viewports['Viewport: 3'].view.setValues(width=0.0102876, 
    height=0.0424793, viewOffsetX=0.000968004, viewOffsetY=0.00124381)
session.viewports['Viewport: 4'].view.setValues(width=0.0100637, 
    height=0.0425167, viewOffsetX=0.00145622, viewOffsetY=0.00142678)
session.viewports['Viewport: 10'].view.setValues(width=0.0109441, 
    height=0.0424424, viewOffsetX=-0.000500632, viewOffsetY=0.00106053)
session.viewports['Viewport: 11'].view.setValues(width=0.0100307, 
    height=0.0423776, viewOffsetX=0.00145146, viewOffsetY=0.00142212)
session.viewports['Viewport: 12'].view.setValues(width=0.0105973, 
    height=0.0424233, viewOffsetX=0.000316103, viewOffsetY=0.000877346)
session.viewports['Viewport: 1'].view.setValues(width=0.0117655, 
    height=0.0450353, viewOffsetX=-0.000737941, viewOffsetY=0.00114239)
session.viewports['Viewport: 2'].view.setValues(width=0.0107804, 
    height=0.0449928, viewOffsetX=0.00135031, viewOffsetY=0.00096159)
session.viewports['Viewport: 3'].view.setValues(width=0.0109284, 
    height=0.0451249, viewOffsetX=0.0010285, viewOffsetY=0.00132432)
session.viewports['Viewport: 4'].view.setValues(width=0.0106905, 
    height=0.0451647, viewOffsetX=0.00150909, viewOffsetY=0.00150468)
session.viewports['Viewport: 10'].view.setValues(width=0.0116257, 
    height=0.0450856, viewOffsetX=-0.000417298, viewOffsetY=0.00114366)
session.viewports['Viewport: 11'].view.setValues(width=0.0106555, 
    height=0.045017, viewOffsetX=0.00150415, viewOffsetY=0.00149976)
session.viewports['Viewport: 12'].view.setValues(width=0.0112572, 
    height=0.0450652, viewOffsetX=0.00038686, viewOffsetY=0.000963139)
session.viewports['Viewport: 1'].view.setValues(width=0.0124971, 
    height=0.0478356, viewOffsetX=-0.000644286, viewOffsetY=0.00123046)
session.viewports['Viewport: 2'].view.setValues(width=0.0114507, 
    height=0.0477904, viewOffsetX=0.00140912, viewOffsetY=0.00105244)
session.viewports['Viewport: 3'].view.setValues(width=0.0116079, 
    height=0.047931, viewOffsetX=0.00109266, viewOffsetY=0.00140971)
session.viewports['Viewport: 4'].view.setValues(width=0.0113553, 
    height=0.0479733, viewOffsetX=0.00156515, viewOffsetY=0.0015873)
session.viewports['Viewport: 10'].view.setValues(width=0.0123485, 
    height=0.047889, viewOffsetX=-0.000328911, viewOffsetY=0.00123183)
session.viewports['Viewport: 11'].view.setValues(width=0.0113181, 
    height=0.0478164, viewOffsetX=0.00156004, viewOffsetY=0.00158211)
session.viewports['Viewport: 12'].view.setValues(width=0.0119571, 
    height=0.0478672, viewOffsetX=0.000461906, viewOffsetY=0.00105413)
session.viewports['Viewport: 1'].view.setValues(width=0.0132729, 
    height=0.0508051, viewOffsetX=-0.000544972, viewOffsetY=0.00132385)
session.viewports['Viewport: 2'].view.setValues(width=0.0121615, 
    height=0.050757, viewOffsetX=0.00147148, viewOffsetY=0.00114877)
session.viewports['Viewport: 3'].view.setValues(width=0.0123285, 
    height=0.0509066, viewOffsetX=0.00116069, viewOffsetY=0.00150027)
session.viewports['Viewport: 4'].view.setValues(width=0.0120602, 
    height=0.0509517, viewOffsetX=0.00162461, viewOffsetY=0.00167491)
session.viewports['Viewport: 10'].view.setValues(width=0.0131151, 
    height=0.0508618, viewOffsetX=-0.000235183, viewOffsetY=0.00132533)
session.viewports['Viewport: 11'].view.setValues(width=0.0120208, 
    height=0.050785, viewOffsetX=0.0016193, viewOffsetY=0.00166943)
session.viewports['Viewport: 12'].view.setValues(width=0.0126994, 
    height=0.0508386, viewOffsetX=0.000541487, viewOffsetY=0.00115062)
session.viewports['Viewport: 1'].view.setValues(width=0.0140954, 
    height=0.0539535, viewOffsetX=-0.000439676, viewOffsetY=0.00142287)
session.viewports['Viewport: 2'].view.setValues(width=0.0129151, 
    height=0.0539022, viewOffsetX=0.0015376, viewOffsetY=0.00125091)
session.viewports['Viewport: 3'].view.setValues(width=0.0130926, 
    height=0.0540613, viewOffsetX=0.00123283, viewOffsetY=0.00159627)
session.viewports['Viewport: 4'].view.setValues(width=0.0128077, 
    height=0.0541094, viewOffsetX=0.00168764, viewOffsetY=0.0017678)
session.viewports['Viewport: 10'].view.setValues(width=0.0139278, 
    height=0.0540137, viewOffsetX=-0.000135811, viewOffsetY=0.00142446)
session.viewports['Viewport: 11'].view.setValues(width=0.0127658, 
    height=0.0539324, viewOffsetX=0.00168212, viewOffsetY=0.00176202)
session.viewports['Viewport: 12'].view.setValues(width=0.0134863, 
    height=0.0539888, viewOffsetX=0.00062586, viewOffsetY=0.00125293)
session.viewports['Viewport: 1'].view.setValues(width=0.013225, 
    height=0.050622, viewOffsetX=-0.000400489, viewOffsetY=0.00129721)
session.viewports['Viewport: 2'].view.setValues(width=0.0121176, 
    height=0.0505737, viewOffsetX=0.0016088, viewOffsetY=0.00112274)
session.viewports['Viewport: 3'].view.setValues(width=0.0122842, 
    height=0.0507234, viewOffsetX=0.00129908, viewOffsetY=0.00147299)
session.viewports['Viewport: 4'].view.setValues(width=0.0120169, 
    height=0.0507687, viewOffsetX=0.00176122, viewOffsetY=0.00164703)
session.viewports['Viewport: 10'].view.setValues(width=0.0130678, 
    height=0.0506785, viewOffsetX=-9.16575e-05, viewOffsetY=0.00129866)
session.viewports['Viewport: 11'].view.setValues(width=0.0119776, 
    height=0.0506025, viewOffsetX=0.00175546, viewOffsetY=0.00164165)
session.viewports['Viewport: 12'].view.setValues(width=0.0126535, 
    height=0.050655, viewOffsetX=0.000682386, viewOffsetY=0.00112455)
session.viewports['Viewport: 1'].view.setValues(width=0.012456, 
    height=0.0476784, viewOffsetX=-0.000365138, viewOffsetY=0.00118391)
session.viewports['Viewport: 2'].view.setValues(width=0.011413, 
    height=0.0476331, viewOffsetX=0.00168172, viewOffsetY=0.00100643)
session.viewports['Viewport: 3'].view.setValues(width=0.0115698, 
    height=0.0477737, viewOffsetX=0.00136619, viewOffsetY=0.00136257)
session.viewports['Viewport: 4'].view.setValues(width=0.0113181, 
    height=0.0478162, viewOffsetX=0.00183692, viewOffsetY=0.00153961)
session.viewports['Viewport: 10'].view.setValues(width=0.012308, 
    height=0.0477316, viewOffsetX=-5.049e-05, viewOffsetY=0.00118523)
session.viewports['Viewport: 11'].view.setValues(width=0.011281, 
    height=0.0476598, viewOffsetX=0.00183091, viewOffsetY=0.00153458)
session.viewports['Viewport: 12'].view.setValues(width=0.0119178, 
    height=0.0477097, viewOffsetX=0.000738069, viewOffsetY=0.00100805)
session.viewports['Viewport: 1'].view.setValues(width=0.011729, 
    height=0.0448956, viewOffsetX=-0.000331743, viewOffsetY=0.00107686)
session.viewports['Viewport: 2'].view.setValues(width=0.0107469, 
    height=0.044853, viewOffsetX=0.00175032, viewOffsetY=0.000896576)
session.viewports['Viewport: 3'].view.setValues(width=0.0108945, 
    height=0.0449852, viewOffsetX=0.00142934, viewOffsetY=0.00125822)
session.viewports['Viewport: 4'].view.setValues(width=0.0106574, 
    height=0.0450251, viewOffsetX=0.00190813, viewOffsetY=0.00143808)
session.viewports['Viewport: 10'].view.setValues(width=0.0115896, 
    height=0.0449457, viewOffsetX=-1.16429e-05, viewOffsetY=0.00107807)
session.viewports['Viewport: 11'].view.setValues(width=0.0106225, 
    height=0.0448777, viewOffsetX=0.00190189, viewOffsetY=0.00143337)
session.viewports['Viewport: 12'].view.setValues(width=0.0112222, 
    height=0.0449251, viewOffsetX=0.000790518, viewOffsetY=0.000898021)
session.viewports['Viewport: 1'].view.setValues(width=0.0133502, 
    height=0.0511014, viewOffsetX=-0.000290675, viewOffsetY=0.00141585)
session.viewports['Viewport: 2'].view.setValues(width=0.0122324, 
    height=0.0510531, viewOffsetX=0.00172717, viewOffsetY=0.00124071)
session.viewports['Viewport: 3'].view.setValues(width=0.0124004, 
    height=0.0512032, viewOffsetX=0.00141607, viewOffsetY=0.00159241)
session.viewports['Viewport: 4'].view.setValues(width=0.0121305, 
    height=0.0512484, viewOffsetX=0.00188005, viewOffsetY=0.0017671)
session.viewports['Viewport: 10'].view.setValues(width=0.0131916, 
    height=0.0511583, viewOffsetX=1.95812e-05, viewOffsetY=0.00141743)
session.viewports['Viewport: 11'].view.setValues(width=0.0120908, 
    height=0.0510808, viewOffsetX=0.0018739, viewOffsetY=0.00176132)
session.viewports['Viewport: 12'].view.setValues(width=0.0127735, 
    height=0.0511351, viewOffsetX=0.00079703, viewOffsetY=0.00124271)
session.viewports['Viewport: 1'].view.setValues(width=0.0151154, 
    height=0.0578578, viewOffsetX=-0.000242501, viewOffsetY=0.00179248)
session.viewports['Viewport: 2'].view.setValues(width=0.0138497, 
    height=0.0578027, viewOffsetX=0.00169139, viewOffsetY=0.00162414)
session.viewports['Viewport: 3'].view.setValues(width=0.01404, 
    height=0.0579735, viewOffsetX=0.00139324, viewOffsetY=0.00196265)
session.viewports['Viewport: 4'].view.setValues(width=0.0137345, 
    height=0.0580251, viewOffsetX=0.0018379, viewOffsetY=0.00213055)
session.viewports['Viewport: 10'].view.setValues(width=0.0149357, 
    height=0.0579223, viewOffsetX=5.48838e-05, viewOffsetY=0.00179448)
session.viewports['Viewport: 11'].view.setValues(width=0.0136896, 
    height=0.0578352, viewOffsetX=0.00183189, viewOffsetY=0.00212358)
session.viewports['Viewport: 12'].view.setValues(width=0.0144622, 
    height=0.0578957, viewOffsetX=0.000800019, viewOffsetY=0.00162675)
session.viewports['Viewport: 1'].view.setValues(width=0.016016, 
    height=0.0613054, viewOffsetX=-0.000216574, viewOffsetY=0.00198761)
session.viewports['Viewport: 2'].view.setValues(width=0.0146748, 
    height=0.0612466, viewOffsetX=0.00166902, viewOffsetY=0.00182319)
session.viewports['Viewport: 3'].view.setValues(width=0.0148767, 
    height=0.0614284, viewOffsetX=0.00137833, viewOffsetY=0.00215407)
session.viewports['Viewport: 4'].view.setValues(width=0.0145531, 
    height=0.0614834, viewOffsetX=0.00181188, viewOffsetY=0.00231804)
session.viewports['Viewport: 10'].view.setValues(width=0.0158257, 
    height=0.0613737, viewOffsetX=7.34059e-05, viewOffsetY=0.00198983)
session.viewports['Viewport: 11'].view.setValues(width=0.0145055, 
    height=0.0612822, viewOffsetX=0.00180595, viewOffsetY=0.00231046)
session.viewports['Viewport: 12'].view.setValues(width=0.0153239, 
    height=0.0613451, viewOffsetX=0.000799951, viewOffsetY=0.00182612)
session.viewports['Viewport: 1'].view.setValues(width=0.0170035, 
    height=0.0650852, viewOffsetX=-0.000156509, viewOffsetY=0.00219169)
session.viewports['Viewport: 2'].view.setValues(width=0.0155796, 
    height=0.0650226, viewOffsetX=0.00168218, viewOffsetY=0.00203106)
session.viewports['Viewport: 3'].view.setValues(width=0.015794, 
    height=0.065216, viewOffsetX=0.00139871, viewOffsetY=0.00235457)
session.viewports['Viewport: 4'].view.setValues(width=0.0154505, 
    height=0.0652747, viewOffsetX=0.00182143, viewOffsetY=0.00251476)
session.viewports['Viewport: 10'].view.setValues(width=0.0168014, 
    height=0.0651578, viewOffsetX=0.000126313, viewOffsetY=0.00219413)
session.viewports['Viewport: 11'].view.setValues(width=0.0153999, 
    height=0.0650611, viewOffsetX=0.00181548, viewOffsetY=0.00250653)
session.viewports['Viewport: 12'].view.setValues(width=0.0162687, 
    height=0.0651272, viewOffsetX=0.000834836, viewOffsetY=0.00203433)
session.viewports['Viewport: 1'].view.setValues(width=0.0180483, 
    height=0.0690842, viewOffsetX=-9.28714e-05, viewOffsetY=0.0024077)
session.viewports['Viewport: 2'].view.setValues(width=0.0165367, 
    height=0.0690175, viewOffsetX=0.00169599, viewOffsetY=0.00225109)
session.viewports['Viewport: 3'].view.setValues(width=0.0167645, 
    height=0.0692233, viewOffsetX=0.0014202, viewOffsetY=0.00256679)
session.viewports['Viewport: 4'].view.setValues(width=0.0163999, 
    height=0.0692858, viewOffsetX=0.00183142, viewOffsetY=0.00272295)
session.viewports['Viewport: 10'].view.setValues(width=0.0178337, 
    height=0.0691612, viewOffsetX=0.000182347, viewOffsetY=0.00241038)
session.viewports['Viewport: 11'].view.setValues(width=0.0163462, 
    height=0.0690591, viewOffsetX=0.00182543, viewOffsetY=0.00271404)
session.viewports['Viewport: 12'].view.setValues(width=0.0172682, 
    height=0.0691285, viewOffsetX=0.000871724, viewOffsetY=0.00225471)
session.viewports['Viewport: 1'].view.setValues(width=0.0191546, 
    height=0.0733189, viewOffsetX=-2.54848e-05, viewOffsetY=0.00263644)
session.viewports['Viewport: 2'].view.setValues(width=0.0175503, 
    height=0.0732478, viewOffsetX=0.00171062, viewOffsetY=0.00248409)
session.viewports['Viewport: 3'].view.setValues(width=0.0177922, 
    height=0.0734668, viewOffsetX=0.00144295, viewOffsetY=0.00279152)
session.viewports['Viewport: 4'].view.setValues(width=0.0174053, 
    height=0.0735334, viewOffsetX=0.00184199, viewOffsetY=0.00294341)
session.viewports['Viewport: 10'].view.setValues(width=0.0189269, 
    height=0.0734006, viewOffsetX=0.000241682, viewOffsetY=0.00263938)
session.viewports['Viewport: 11'].view.setValues(width=0.0173484, 
    height=0.0732928, viewOffsetX=0.00183597, viewOffsetY=0.00293378)
session.viewports['Viewport: 12'].view.setValues(width=0.0183266, 
    height=0.0733657, viewOffsetX=0.000910787, viewOffsetY=0.00248809)
session.viewports['Viewport: 1'].view.setValues(width=0.0203257, 
    height=0.0778017, viewOffsetX=4.58516e-05, viewOffsetY=0.00287859)
session.viewports['Viewport: 2'].view.setValues(width=0.0186233, 
    height=0.0777261, viewOffsetX=0.0017261, viewOffsetY=0.00273074)
session.viewports['Viewport: 3'].view.setValues(width=0.0188801, 
    height=0.0779591, viewOffsetX=0.00146704, viewOffsetY=0.00302941)
session.viewports['Viewport: 4'].view.setValues(width=0.0184697, 
    height=0.0780301, viewOffsetX=0.00185318, viewOffsetY=0.0031768)
session.viewports['Viewport: 10'].view.setValues(width=0.0200841, 
    height=0.0778885, viewOffsetX=0.000304495, viewOffsetY=0.0028818)
session.viewports['Viewport: 11'].view.setValues(width=0.0184092, 
    height=0.0777747, viewOffsetX=0.00184712, viewOffsetY=0.00316641)
session.viewports['Viewport: 12'].view.setValues(width=0.019447, 
    height=0.0778511, viewOffsetX=0.000952139, viewOffsetY=0.00273514)
session.viewports['Viewport: 12'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM10', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 12'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM9', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 12'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM8', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 12'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM2', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 12'].odbDisplay.setFrame(step=1, frame=101 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 2'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 4'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 10'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 3'].odbDisplay.setFrame(step=1, frame=101 )
session.viewports['Viewport: 11'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 12'].odbDisplay.setFrame(step=1, frame=100 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=101 )
session.viewports['Viewport: 2'].odbDisplay.setFrame(step=1, frame=101 )
session.viewports['Viewport: 4'].odbDisplay.setFrame(step=1, frame=101 )
session.viewports['Viewport: 10'].odbDisplay.setFrame(step=1, frame=101 )
session.viewports['Viewport: 3'].odbDisplay.setFrame(step=1, frame=100 )
session.viewports['Viewport: 11'].odbDisplay.setFrame(step=1, frame=101 )
session.viewports['Viewport: 12'].odbDisplay.setFrame(step=1, frame=99 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=100 )
session.viewports['Viewport: 2'].odbDisplay.setFrame(step=1, frame=100 )
session.viewports['Viewport: 4'].odbDisplay.setFrame(step=1, frame=100 )
session.viewports['Viewport: 10'].odbDisplay.setFrame(step=1, frame=100 )
session.viewports['Viewport: 3'].odbDisplay.setFrame(step=1, frame=99 )
session.viewports['Viewport: 11'].odbDisplay.setFrame(step=1, frame=100 )
session.viewports['Viewport: 12'].odbDisplay.setFrame(step=1, frame=98 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=99 )
session.viewports['Viewport: 2'].odbDisplay.setFrame(step=1, frame=99 )
session.viewports['Viewport: 4'].odbDisplay.setFrame(step=1, frame=99 )
session.viewports['Viewport: 10'].odbDisplay.setFrame(step=1, frame=99 )
session.viewports['Viewport: 3'].odbDisplay.setFrame(step=1, frame=98 )
session.viewports['Viewport: 11'].odbDisplay.setFrame(step=1, frame=99 )
session.viewports['Viewport: 12'].odbDisplay.setFrame(step=1, frame=99 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=100 )
session.viewports['Viewport: 2'].odbDisplay.setFrame(step=1, frame=100 )
session.viewports['Viewport: 4'].odbDisplay.setFrame(step=1, frame=100 )
session.viewports['Viewport: 10'].odbDisplay.setFrame(step=1, frame=100 )
session.viewports['Viewport: 3'].odbDisplay.setFrame(step=1, frame=99 )
session.viewports['Viewport: 11'].odbDisplay.setFrame(step=1, frame=100 )
session.viewports['Viewport: 12'].odbDisplay.setFrame(step=1, frame=100 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=101 )
session.viewports['Viewport: 2'].odbDisplay.setFrame(step=1, frame=101 )
session.viewports['Viewport: 4'].odbDisplay.setFrame(step=1, frame=101 )
session.viewports['Viewport: 10'].odbDisplay.setFrame(step=1, frame=101 )
session.viewports['Viewport: 3'].odbDisplay.setFrame(step=1, frame=100 )
session.viewports['Viewport: 11'].odbDisplay.setFrame(step=1, frame=101 )
session.viewports['Viewport: 12'].odbDisplay.setFrame(step=1, frame=101 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 2'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 4'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 10'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 3'].odbDisplay.setFrame(step=1, frame=101 )
session.viewports['Viewport: 11'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 12'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=103 )
session.viewports['Viewport: 2'].odbDisplay.setFrame(step=1, frame=103 )
session.viewports['Viewport: 4'].odbDisplay.setFrame(step=1, frame=103 )
session.viewports['Viewport: 10'].odbDisplay.setFrame(step=1, frame=103 )
session.viewports['Viewport: 3'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 11'].odbDisplay.setFrame(step=1, frame=103 )
session.viewports['Viewport: 12'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=103 )
session.viewports['Viewport: 2'].odbDisplay.setFrame(step=1, frame=103 )
session.viewports['Viewport: 4'].odbDisplay.setFrame(step=1, frame=103 )
session.viewports['Viewport: 10'].odbDisplay.setFrame(step=1, frame=103 )
session.viewports['Viewport: 3'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 11'].odbDisplay.setFrame(step=1, frame=103 )
session.viewports['Viewport: 12'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=103 )
session.viewports['Viewport: 2'].odbDisplay.setFrame(step=1, frame=103 )
session.viewports['Viewport: 4'].odbDisplay.setFrame(step=1, frame=103 )
session.viewports['Viewport: 10'].odbDisplay.setFrame(step=1, frame=103 )
session.viewports['Viewport: 3'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 11'].odbDisplay.setFrame(step=1, frame=103 )
session.viewports['Viewport: 12'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=103 )
session.viewports['Viewport: 2'].odbDisplay.setFrame(step=1, frame=103 )
session.viewports['Viewport: 4'].odbDisplay.setFrame(step=1, frame=103 )
session.viewports['Viewport: 10'].odbDisplay.setFrame(step=1, frame=103 )
session.viewports['Viewport: 3'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 11'].odbDisplay.setFrame(step=1, frame=103 )
session.viewports['Viewport: 12'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=103 )
session.viewports['Viewport: 2'].odbDisplay.setFrame(step=1, frame=103 )
session.viewports['Viewport: 4'].odbDisplay.setFrame(step=1, frame=103 )
session.viewports['Viewport: 10'].odbDisplay.setFrame(step=1, frame=103 )
session.viewports['Viewport: 3'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 11'].odbDisplay.setFrame(step=1, frame=103 )
session.viewports['Viewport: 12'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=103 )
session.viewports['Viewport: 2'].odbDisplay.setFrame(step=1, frame=103 )
session.viewports['Viewport: 4'].odbDisplay.setFrame(step=1, frame=103 )
session.viewports['Viewport: 10'].odbDisplay.setFrame(step=1, frame=103 )
session.viewports['Viewport: 3'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 11'].odbDisplay.setFrame(step=1, frame=103 )
session.viewports['Viewport: 12'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=103 )
session.viewports['Viewport: 2'].odbDisplay.setFrame(step=1, frame=103 )
session.viewports['Viewport: 4'].odbDisplay.setFrame(step=1, frame=103 )
session.viewports['Viewport: 10'].odbDisplay.setFrame(step=1, frame=103 )
session.viewports['Viewport: 3'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 11'].odbDisplay.setFrame(step=1, frame=103 )
session.viewports['Viewport: 12'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM8', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 2'].makeCurrent()
session.viewports['Viewport: 2'].setValues(origin=(67.4114532470703, 
    11.3923645019531), width=38.296875)
session.viewports['Viewport: 2'].setValues(width=33.1458320617676)
session.viewports['Viewport: 2'].setValues(width=38.5208320617676)
session.viewports['Viewport: 2'].setValues(origin=(76.59375, 11.3923645019531), 
    width=29.1145820617676)
#* RangeError: width must be a Float in the range: 30 <= width <= 3350
session.viewports['Viewport: 2'].setValues(origin=(71.6666641235352, 
    11.839111328125))
session.viewports['Viewport: 2'].setValues(width=33.59375)
session.viewports['Viewport: 2'].setValues(width=34.0416641235352)
session.viewports['Viewport: 2'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM9', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 2'].odbDisplay.setFrame(step=1, frame=0 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=0 )
session.viewports['Viewport: 4'].odbDisplay.setFrame(step=1, frame=0 )
session.viewports['Viewport: 10'].odbDisplay.setFrame(step=1, frame=0 )
session.viewports['Viewport: 12'].odbDisplay.setFrame(step=1, frame=0 )
session.viewports['Viewport: 3'].odbDisplay.setFrame(step=1, frame=0 )
session.viewports['Viewport: 11'].odbDisplay.setFrame(step=1, frame=0 )
session.viewports['Viewport: 2'].odbDisplay.setFrame(step=1, frame=1 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=1 )
session.viewports['Viewport: 4'].odbDisplay.setFrame(step=1, frame=1 )
session.viewports['Viewport: 10'].odbDisplay.setFrame(step=1, frame=1 )
session.viewports['Viewport: 12'].odbDisplay.setFrame(step=1, frame=1 )
session.viewports['Viewport: 3'].odbDisplay.setFrame(step=1, frame=1 )
session.viewports['Viewport: 11'].odbDisplay.setFrame(step=1, frame=1 )
session.viewports['Viewport: 2'].odbDisplay.setFrame(step=1, frame=2 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=2 )
session.viewports['Viewport: 4'].odbDisplay.setFrame(step=1, frame=2 )
session.viewports['Viewport: 10'].odbDisplay.setFrame(step=1, frame=2 )
session.viewports['Viewport: 12'].odbDisplay.setFrame(step=1, frame=2 )
session.viewports['Viewport: 3'].odbDisplay.setFrame(step=1, frame=2 )
session.viewports['Viewport: 11'].odbDisplay.setFrame(step=1, frame=2 )
session.viewports['Viewport: 2'].odbDisplay.setFrame(step=1, frame=3 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=3 )
session.viewports['Viewport: 4'].odbDisplay.setFrame(step=1, frame=3 )
session.viewports['Viewport: 10'].odbDisplay.setFrame(step=1, frame=3 )
session.viewports['Viewport: 12'].odbDisplay.setFrame(step=1, frame=3 )
session.viewports['Viewport: 3'].odbDisplay.setFrame(step=1, frame=3 )
session.viewports['Viewport: 11'].odbDisplay.setFrame(step=1, frame=3 )
session.viewports['Viewport: 2'].odbDisplay.setFrame(step=1, frame=4 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=4 )
session.viewports['Viewport: 4'].odbDisplay.setFrame(step=1, frame=4 )
session.viewports['Viewport: 10'].odbDisplay.setFrame(step=1, frame=4 )
session.viewports['Viewport: 12'].odbDisplay.setFrame(step=1, frame=4 )
session.viewports['Viewport: 3'].odbDisplay.setFrame(step=1, frame=4 )
session.viewports['Viewport: 11'].odbDisplay.setFrame(step=1, frame=4 )
session.viewports['Viewport: 2'].odbDisplay.setFrame(step=1, frame=5 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=5 )
session.viewports['Viewport: 4'].odbDisplay.setFrame(step=1, frame=5 )
session.viewports['Viewport: 10'].odbDisplay.setFrame(step=1, frame=5 )
session.viewports['Viewport: 12'].odbDisplay.setFrame(step=1, frame=5 )
session.viewports['Viewport: 3'].odbDisplay.setFrame(step=1, frame=5 )
session.viewports['Viewport: 11'].odbDisplay.setFrame(step=1, frame=5 )
session.viewports['Viewport: 2'].odbDisplay.setFrame(step=1, frame=6 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=6 )
session.viewports['Viewport: 4'].odbDisplay.setFrame(step=1, frame=6 )
session.viewports['Viewport: 10'].odbDisplay.setFrame(step=1, frame=6 )
session.viewports['Viewport: 12'].odbDisplay.setFrame(step=1, frame=6 )
session.viewports['Viewport: 3'].odbDisplay.setFrame(step=1, frame=6 )
session.viewports['Viewport: 11'].odbDisplay.setFrame(step=1, frame=6 )
session.viewports['Viewport: 2'].odbDisplay.setFrame(step=1, frame=7 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=7 )
session.viewports['Viewport: 4'].odbDisplay.setFrame(step=1, frame=7 )
session.viewports['Viewport: 10'].odbDisplay.setFrame(step=1, frame=7 )
session.viewports['Viewport: 12'].odbDisplay.setFrame(step=1, frame=7 )
session.viewports['Viewport: 3'].odbDisplay.setFrame(step=1, frame=7 )
session.viewports['Viewport: 11'].odbDisplay.setFrame(step=1, frame=7 )
session.viewports['Viewport: 2'].odbDisplay.setFrame(step=1, frame=8 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=8 )
session.viewports['Viewport: 4'].odbDisplay.setFrame(step=1, frame=8 )
session.viewports['Viewport: 10'].odbDisplay.setFrame(step=1, frame=8 )
session.viewports['Viewport: 12'].odbDisplay.setFrame(step=1, frame=8 )
session.viewports['Viewport: 3'].odbDisplay.setFrame(step=1, frame=8 )
session.viewports['Viewport: 11'].odbDisplay.setFrame(step=1, frame=8 )
session.viewports['Viewport: 2'].odbDisplay.setFrame(step=1, frame=9 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=9 )
session.viewports['Viewport: 4'].odbDisplay.setFrame(step=1, frame=9 )
session.viewports['Viewport: 10'].odbDisplay.setFrame(step=1, frame=9 )
session.viewports['Viewport: 12'].odbDisplay.setFrame(step=1, frame=9 )
session.viewports['Viewport: 3'].odbDisplay.setFrame(step=1, frame=9 )
session.viewports['Viewport: 11'].odbDisplay.setFrame(step=1, frame=9 )
session.viewports['Viewport: 2'].odbDisplay.setFrame(step=1, frame=10 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=10 )
session.viewports['Viewport: 4'].odbDisplay.setFrame(step=1, frame=10 )
session.viewports['Viewport: 10'].odbDisplay.setFrame(step=1, frame=10 )
session.viewports['Viewport: 12'].odbDisplay.setFrame(step=1, frame=10 )
session.viewports['Viewport: 3'].odbDisplay.setFrame(step=1, frame=10 )
session.viewports['Viewport: 11'].odbDisplay.setFrame(step=1, frame=10 )
session.viewports['Viewport: 2'].odbDisplay.setFrame(step=1, frame=11 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=11 )
session.viewports['Viewport: 4'].odbDisplay.setFrame(step=1, frame=11 )
session.viewports['Viewport: 10'].odbDisplay.setFrame(step=1, frame=11 )
session.viewports['Viewport: 12'].odbDisplay.setFrame(step=1, frame=11 )
session.viewports['Viewport: 3'].odbDisplay.setFrame(step=1, frame=11 )
session.viewports['Viewport: 11'].odbDisplay.setFrame(step=1, frame=11 )
session.viewports['Viewport: 2'].odbDisplay.setFrame(step=1, frame=12 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=12 )
session.viewports['Viewport: 4'].odbDisplay.setFrame(step=1, frame=12 )
session.viewports['Viewport: 10'].odbDisplay.setFrame(step=1, frame=12 )
session.viewports['Viewport: 12'].odbDisplay.setFrame(step=1, frame=12 )
session.viewports['Viewport: 3'].odbDisplay.setFrame(step=1, frame=12 )
session.viewports['Viewport: 11'].odbDisplay.setFrame(step=1, frame=12 )
session.viewports[session.currentViewportName].odbDisplay.setFrame(
    step='step2_insitu', frame=39)
session.viewports[session.currentViewportName].odbDisplay.setFrame(
    step='step2_insitu', frame=53)
session.viewports[session.currentViewportName].odbDisplay.setFrame(
    step='step2_insitu', frame=103)
session.viewports['Viewport: 4'].makeCurrent()
session.viewports['Viewport: 4'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM8', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 4'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM9', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 4'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM10', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 10'].makeCurrent()
session.viewports['Viewport: 10'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM8', outputPosition=INTEGRATION_POINT, )
session. linkedViewportCommands.setValues(linkViewports=False)
session. linkedViewportCommands.setValues(linkViewports=True)
session.viewports['Viewport: 10'].view.setValues(session.views['Back'])
session.viewports['Viewport: 10'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM9', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM8', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 11'].makeCurrent()
session.viewports['Viewport: 10'].makeCurrent()
session.viewports['Viewport: 10'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM1', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 10'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM2', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 10'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM1', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 10'].view.setValues(session.views['Back'])
session.viewports['Viewport: 1'].view.setValues(width=0.0633896, 
    height=0.242639, viewOffsetX=7.88207e-05, viewOffsetY=-0.000222749)
session.viewports['Viewport: 2'].view.setValues(width=0.0583029, 
    height=0.243332, viewOffsetX=0.000250737, viewOffsetY=-0.000236926)
session.viewports['Viewport: 3'].view.setValues(width=0.0586965, 
    height=0.242367, viewOffsetX=0.000222699, viewOffsetY=-0.000209057)
session.viewports['Viewport: 4'].view.setValues(width=0.0573415, 
    height=0.242255, viewOffsetX=0.000261436, viewOffsetY=-0.00019557)
session.viewports['Viewport: 10'].view.setValues(width=0.0625604, 
    height=0.242616, viewOffsetX=0.000105085, viewOffsetY=-0.000222728)
session.viewports['Viewport: 11'].view.setValues(width=0.0573276, 
    height=0.242196, viewOffsetX=0.000261372, viewOffsetY=-0.000195522)
session.viewports['Viewport: 12'].view.setValues(width=0.060565, 
    height=0.242456, viewOffsetX=0.000170939, viewOffsetY=-0.000236073)
session.viewports['Viewport: 1'].view.setValues(width=0.0521642, 
    height=0.199671, viewOffsetX=0.000133866, viewOffsetY=-0.000378306)
session.viewports['Viewport: 2'].view.setValues(width=0.0480513, 
    height=0.200546, viewOffsetX=0.000426488, viewOffsetY=-0.000402997)
session.viewports['Viewport: 3'].view.setValues(width=0.0483916, 
    height=0.199817, viewOffsetX=0.000378923, viewOffsetY=-0.000355711)
session.viewports['Viewport: 4'].view.setValues(width=0.0473188, 
    height=0.199911, viewOffsetX=0.000445249, viewOffsetY=-0.000333074)
session.viewports['Viewport: 10'].view.setValues(width=0.0514956, 
    height=0.199706, viewOffsetX=0.000178519, viewOffsetY=-0.000378372)
session.viewports['Viewport: 11'].view.setValues(width=0.0472987, 
    height=0.199826, viewOffsetX=0.00044506, viewOffsetY=-0.000332932)
session.viewports['Viewport: 12'].view.setValues(width=0.0498748, 
    height=0.199661, viewOffsetX=0.000290519, viewOffsetY=-0.000401217)
session.viewports['Viewport: 1'].view.setValues(width=0.0504452, 
    height=0.193091, viewOffsetX=0.000200443, viewOffsetY=-0.000566454)
session.viewports['Viewport: 2'].view.setValues(width=0.0464589, height=0.1939, 
    viewOffsetX=0.000638474, viewOffsetY=-0.000603307)
session.viewports['Viewport: 3'].view.setValues(width=0.0467811, 
    height=0.193167, viewOffsetX=0.000567184, viewOffsetY=-0.00053244)
session.viewports['Viewport: 4'].view.setValues(width=0.045735, height=0.19322, 
    viewOffsetX=0.000666335, viewOffsetY=-0.000498459)
session.viewports['Viewport: 10'].view.setValues(width=0.0497965, 
    height=0.193116, viewOffsetX=0.000267292, viewOffsetY=-0.000566528)
session.viewports['Viewport: 11'].view.setValues(width=0.045717, 
    height=0.193144, viewOffsetX=0.000666071, viewOffsetY=-0.000498262)
session.viewports['Viewport: 12'].view.setValues(width=0.0482282, 
    height=0.193069, viewOffsetX=0.000434978, viewOffsetY=-0.00060072)
session.viewports['Viewport: 1'].view.setValues(width=0.0476215, 
    height=0.182283, viewOffsetX=0.000260516, viewOffsetY=-0.000736221)
session.viewports['Viewport: 2'].view.setValues(width=0.0438598, 
    height=0.183052, viewOffsetX=0.000829853, viewOffsetY=-0.000784144)
session.viewports['Viewport: 3'].view.setValues(width=0.0441642, 
    height=0.182361, viewOffsetX=0.000737197, viewOffsetY=-0.000692038)
session.viewports['Viewport: 4'].view.setValues(width=0.0431774, 
    height=0.182415, viewOffsetX=0.000866082, viewOffsetY=-0.000647883)
session.viewports['Viewport: 10'].view.setValues(width=0.0470094, 
    height=0.182308, viewOffsetX=0.000347401, viewOffsetY=-0.000736321)
session.viewports['Viewport: 11'].view.setValues(width=0.0431602, 
    height=0.182342, viewOffsetX=0.000865737, viewOffsetY=-0.000647624)
session.viewports['Viewport: 12'].view.setValues(width=0.0455293, 
    height=0.182265, viewOffsetX=0.00056535, viewOffsetY=-0.000780769)
session.viewports['Viewport: 1'].view.setValues(width=0.0450778, 
    height=0.172546, viewOffsetX=0.000168922, viewOffsetY=-0.000713425)
session.viewports['Viewport: 2'].view.setValues(width=0.0415173, 
    height=0.173276, viewOffsetX=0.000863864, viewOffsetY=-0.000770896)
session.viewports['Viewport: 3'].view.setValues(width=0.0418044, 
    height=0.172617, viewOffsetX=0.000751679, viewOffsetY=-0.00065965)
session.viewports['Viewport: 4'].view.setValues(width=0.0408698, 
    height=0.172666, viewOffsetX=0.00090939, viewOffsetY=-0.000605938)
session.viewports['Viewport: 10'].view.setValues(width=0.0444984, 
    height=0.17257, viewOffsetX=0.000275089, viewOffsetY=-0.000713521)
session.viewports['Viewport: 11'].view.setValues(width=0.0408536, 
    height=0.172597, viewOffsetX=0.000909027, viewOffsetY=-0.000605696)
session.viewports['Viewport: 12'].view.setValues(width=0.0430978, 
    height=0.172531, viewOffsetX=0.000541245, viewOffsetY=-0.00076758)
session.viewports['Viewport: 1'].view.setValues(width=0.0426387, 
    height=0.16321, viewOffsetX=8.16161e-05, viewOffsetY=-0.000691455)
session.viewports['Viewport: 2'].view.setValues(width=0.0392712, 
    height=0.163902, viewOffsetX=0.000895953, viewOffsetY=-0.000758003)
session.viewports['Viewport: 3'].view.setValues(width=0.0395421, 
    height=0.163276, viewOffsetX=0.000765208, viewOffsetY=-0.000628569)
session.viewports['Viewport: 4'].view.setValues(width=0.0386577, 
    height=0.16332, viewOffsetX=0.000950323, viewOffsetY=-0.000565775)
session.viewports['Viewport: 10'].view.setValues(width=0.0420906, 
    height=0.163232, viewOffsetX=0.000206112, viewOffsetY=-0.000691548)
session.viewports['Viewport: 11'].view.setValues(width=0.0386423, 
    height=0.163255, viewOffsetX=0.000949944, viewOffsetY=-0.000565549)
session.viewports['Viewport: 12'].view.setValues(width=0.0407662, 
    height=0.163197, viewOffsetX=0.00051809, viewOffsetY=-0.000754742)
session.viewports['Viewport: 1'].view.setValues(width=0.0403197, 
    height=0.154334, viewOffsetX=-1.45473e-06, viewOffsetY=-0.000670582)
session.viewports['Viewport: 2'].view.setValues(width=0.0371357, 
    height=0.154989, viewOffsetX=0.000926528, viewOffsetY=-0.000745769)
session.viewports['Viewport: 3'].view.setValues(width=0.0373912, 
    height=0.154394, viewOffsetX=0.000778115, viewOffsetY=-0.000599023)
session.viewports['Viewport: 4'].view.setValues(width=0.0365546, 
    height=0.154435, viewOffsetX=0.000989313, viewOffsetY=-0.000527586)
session.viewports['Viewport: 10'].view.setValues(width=0.0398014, 
    height=0.154354, viewOffsetX=0.000140487, viewOffsetY=-0.000670672)
session.viewports['Viewport: 11'].view.setValues(width=0.03654, 
    height=0.154373, viewOffsetX=0.000988919, viewOffsetY=-0.000527375)
session.viewports['Viewport: 12'].view.setValues(width=0.0385494, 
    height=0.154322, viewOffsetX=0.00049608, viewOffsetY=-0.00074256)
session.viewports['Viewport: 1'].view.setValues(width=0.0381144, 
    height=0.145892, viewOffsetX=-8.0451e-05, viewOffsetY=-0.000650731)
session.viewports['Viewport: 2'].view.setValues(width=0.0351049, 
    height=0.146513, viewOffsetX=0.000955603, viewOffsetY=-0.000734133)
session.viewports['Viewport: 3'].view.setValues(width=0.0353458, 
    height=0.145948, viewOffsetX=0.000790389, viewOffsetY=-0.000570926)
session.viewports['Viewport: 4'].view.setValues(width=0.0345546, 
    height=0.145985, viewOffsetX=0.00102639, viewOffsetY=-0.00049127)
session.viewports['Viewport: 10'].view.setValues(width=0.0376245, 
    height=0.145912, viewOffsetX=7.80809e-05, viewOffsetY=-0.000650818)
session.viewports['Viewport: 11'].view.setValues(width=0.0345409, 
    height=0.145927, viewOffsetX=0.00102598, viewOffsetY=-0.000491073)
session.viewports['Viewport: 12'].view.setValues(width=0.0364413, 
    height=0.145883, viewOffsetX=0.000475149, viewOffsetY=-0.000730975)
session.viewports['Viewport: 1'].view.setValues(width=0.0360188, 
    height=0.137871, viewOffsetX=-0.000155526, viewOffsetY=-0.000631869)
session.viewports['Viewport: 2'].view.setValues(width=0.033175, 
    height=0.138459, viewOffsetX=0.000983238, viewOffsetY=-0.000723078)
session.viewports['Viewport: 3'].view.setValues(width=0.0334021, 
    height=0.137923, viewOffsetX=0.000802055, viewOffsetY=-0.000544226)
session.viewports['Viewport: 4'].view.setValues(width=0.0326542, 
    height=0.137956, viewOffsetX=0.00106162, viewOffsetY=-0.00045676)
session.viewports['Viewport: 10'].view.setValues(width=0.0355558, 
    height=0.137889, viewOffsetX=1.87734e-05, viewOffsetY=-0.000631953)
session.viewports['Viewport: 11'].view.setValues(width=0.0326412, 
    height=0.137902, viewOffsetX=0.0010612, viewOffsetY=-0.000456577)
session.viewports['Viewport: 12'].view.setValues(width=0.0344379, 
    height=0.137863, viewOffsetX=0.000455259, viewOffsetY=-0.000719967)
session.viewports['Viewport: 1'].view.setValues(width=0.0340284, 
    height=0.130252, viewOffsetX=-0.000170414, viewOffsetY=-0.000642091)
session.viewports['Viewport: 2'].view.setValues(width=0.0313421, 
    height=0.130809, viewOffsetX=0.00106624, viewOffsetY=-0.000740883)
session.viewports['Viewport: 3'].view.setValues(width=0.0315561, height=0.1303, 
    viewOffsetX=0.00086948)
session.viewports['Viewport: 4'].view.setValues(width=0.0308493, 
    height=0.130331, viewOffsetX=0.00115135, viewOffsetY=-0.000452043)
session.viewports['Viewport: 10'].view.setValues(width=0.033591, 
    height=0.13027, viewOffsetX=1.8868e-05, viewOffsetY=-0.000642177)
session.viewports['Viewport: 11'].view.setValues(width=0.030837, 
    height=0.130279, viewOffsetX=0.0011509, viewOffsetY=-0.000451862)
session.viewports['Viewport: 12'].view.setValues(width=0.0325352, 
    height=0.130246, viewOffsetX=0.000492877, viewOffsetY=-0.000737695)
session.viewports['Viewport: 1'].view.setValues(width=0.0321392, 
    height=0.123021, viewOffsetX=-0.000184548, viewOffsetY=-0.000651795)
session.viewports['Viewport: 2'].view.setValues(width=0.0296022, 
    height=0.123547, viewOffsetX=0.00114504, viewOffsetY=-0.000757786)
session.viewports['Viewport: 3'].view.setValues(width=0.0298039, 
    height=0.123065, viewOffsetX=0.000933483)
session.viewports['Viewport: 4'].view.setValues(width=0.0291361, 
    height=0.123093, viewOffsetX=0.00123653, viewOffsetY=-0.000447567)
session.viewports['Viewport: 10'].view.setValues(width=0.0317261, 
    height=0.123037, viewOffsetX=1.89579e-05, viewOffsetY=-0.000651882)
session.viewports['Viewport: 11'].view.setValues(width=0.0291246, 
    height=0.123045, viewOffsetX=0.00123604, viewOffsetY=-0.000447388)
session.viewports['Viewport: 12'].view.setValues(width=0.0307291, 
    height=0.123016, viewOffsetX=0.000528588, viewOffsetY=-0.000754526)
session.viewports['Viewport: 1'].view.setValues(width=0.0303469, 
    height=0.11616, viewOffsetX=-0.000197957, viewOffsetY=-0.000661004)
session.viewports['Viewport: 2'].view.setValues(width=0.0279516, 
    height=0.116658, viewOffsetX=0.0012198, viewOffsetY=-0.000773825)
session.viewports['Viewport: 3'].view.setValues(width=0.0281417, 
    height=0.116202, viewOffsetX=0.000994207)
session.viewports['Viewport: 4'].view.setValues(width=0.0275109, 
    height=0.116227, viewOffsetX=0.00131734, viewOffsetY=-0.000443321)
session.viewports['Viewport: 10'].view.setValues(width=0.0299568, 
    height=0.116176, viewOffsetX=1.90431e-05, viewOffsetY=-0.000661092)
session.viewports['Viewport: 11'].view.setValues(width=0.0275, height=0.116181, 
    viewOffsetX=0.00131681, viewOffsetY=-0.000443144)
session.viewports['Viewport: 12'].view.setValues(width=0.0290156, 
    height=0.116156, viewOffsetX=0.00056247, viewOffsetY=-0.000770495)
session.viewports['Viewport: 1'].view.setValues(width=0.0286474, 
    height=0.109655, viewOffsetX=-0.000210673, viewOffsetY=-0.000669737)
session.viewports['Viewport: 2'].view.setValues(width=0.0263864, 
    height=0.110126, viewOffsetX=0.00129069, viewOffsetY=-0.000789036)
session.viewports['Viewport: 3'].view.setValues(width=0.0265655, 
    height=0.109693, viewOffsetX=0.00105179)
session.viewports['Viewport: 4'].view.setValues(width=0.0259699, 
    height=0.109717, viewOffsetX=0.00139397, viewOffsetY=-0.000439296)
session.viewports['Viewport: 10'].view.setValues(width=0.0282792, 
    height=0.10967, viewOffsetX=1.9124e-05, viewOffsetY=-0.000669826)
session.viewports['Viewport: 11'].view.setValues(width=0.0259596, 
    height=0.109673, viewOffsetX=0.00139341, viewOffsetY=-0.00043912)
session.viewports['Viewport: 12'].view.setValues(width=0.0273909, 
    height=0.109652, viewOffsetX=0.0005946, viewOffsetY=-0.00078564)
session.viewports['Viewport: 1'].view.setValues(width=0.0270367, 
    height=0.10349, viewOffsetX=-0.000222725, viewOffsetY=-0.000622129)
session.viewports['Viewport: 2'].view.setValues(width=0.024903, 
    height=0.103935, viewOffsetX=0.00135788, viewOffsetY=-0.000747232)
session.viewports['Viewport: 3'].view.setValues(width=0.0250717, 
    height=0.103525, viewOffsetX=0.00110637, viewOffsetY=-0.000500799)
session.viewports['Viewport: 4'].view.setValues(width=0.0245094, 
    height=0.103547, viewOffsetX=0.00146659, viewOffsetY=-0.000379753)
session.viewports['Viewport: 10'].view.setValues(width=0.0266892, 
    height=0.103504, viewOffsetX=1.92007e-05, viewOffsetY=-0.000622212)
session.viewports['Viewport: 11'].view.setValues(width=0.0244997, 
    height=0.103506, viewOffsetX=0.00146601, viewOffsetY=-0.000379601)
session.viewports['Viewport: 12'].view.setValues(width=0.025851, 
    height=0.103488, viewOffsetX=0.000625053, viewOffsetY=-0.000744017)
session.viewports['Viewport: 1'].view.setValues(width=0.0255109, 
    height=0.0976491, viewOffsetX=-0.000234143, viewOffsetY=-0.000577029)
session.viewports['Viewport: 2'].view.setValues(width=0.0234977, 
    height=0.0980696, viewOffsetX=0.00142154, viewOffsetY=-0.000707631)
session.viewports['Viewport: 3'].view.setValues(width=0.0236566, 
    height=0.0976819, viewOffsetX=0.00115807, viewOffsetY=-0.000450034)
session.viewports['Viewport: 4'].view.setValues(width=0.0231259, 
    height=0.0977018, viewOffsetX=0.0015354, viewOffsetY=-0.000323348)
session.viewports['Viewport: 10'].view.setValues(width=0.0251829, 
    height=0.0976623, viewOffsetX=1.92734e-05, viewOffsetY=-0.000577106)
session.viewports['Viewport: 11'].view.setValues(width=0.0231168, 
    height=0.097663, viewOffsetX=0.00153479, viewOffsetY=-0.000323218)
session.viewports['Viewport: 12'].view.setValues(width=0.0243922, 
    height=0.0976476, viewOffsetX=0.000653905, viewOffsetY=-0.000704586)
session.viewports['Viewport: 1'].view.setValues(width=0.024066, 
    height=0.0921185, viewOffsetX=-0.000284855, viewOffsetY=-0.000594017)
session.viewports['Viewport: 2'].view.setValues(width=0.022167, 
    height=0.0925158, viewOffsetX=0.00144169, viewOffsetY=-0.000730184)
session.viewports['Viewport: 3'].view.setValues(width=0.0223166, 
    height=0.092149, viewOffsetX=0.00116719, viewOffsetY=-0.000461577)
session.viewports['Viewport: 4'].view.setValues(width=0.0218159, 
    height=0.0921672, viewOffsetX=0.00156077, viewOffsetY=-0.000329462)
session.viewports['Viewport: 10'].view.setValues(width=0.0237567, 
    height=0.092131, viewOffsetX=-2.05625e-05, viewOffsetY=-0.000594097)
session.viewports['Viewport: 11'].view.setValues(width=0.0218072, 
    height=0.0921306, viewOffsetX=0.00156015, viewOffsetY=-0.00032933)
session.viewports['Viewport: 12'].view.setValues(width=0.0230108, 
    height=0.0921177, viewOffsetX=0.000641261, viewOffsetY=-0.000727042)
session.viewports['Viewport: 1'].view.setValues(width=0.0226984, 
    height=0.0868837, viewOffsetX=-0.000332857, viewOffsetY=-0.000610099)
session.viewports['Viewport: 2'].view.setValues(width=0.0209074, 
    height=0.0872589, viewOffsetX=0.00146076, viewOffsetY=-0.000751534)
session.viewports['Viewport: 3'].view.setValues(width=0.0210483, 
    height=0.086912, viewOffsetX=0.00117582, viewOffsetY=-0.000472503)
session.viewports['Viewport: 4'].view.setValues(width=0.020576, 
    height=0.0869287, viewOffsetX=0.00158478, viewOffsetY=-0.000335249)
session.viewports['Viewport: 10'].view.setValues(width=0.0224066, 
    height=0.0868954, viewOffsetX=-5.82691e-05, viewOffsetY=-0.00061018)
session.viewports['Viewport: 11'].view.setValues(width=0.0205678, 
    height=0.0868942, viewOffsetX=0.00158415, viewOffsetY=-0.000335115)
session.viewports['Viewport: 12'].view.setValues(width=0.0217033, 
    height=0.0868834, viewOffsetX=0.000629293, viewOffsetY=-0.0007483)
session.viewports['Viewport: 1'].view.setValues(width=0.0214045, 
    height=0.0819308, viewOffsetX=-0.000378275, viewOffsetY=-0.000625315)
session.viewports['Viewport: 2'].view.setValues(width=0.0197157, 
    height=0.082285, viewOffsetX=0.00147881, viewOffsetY=-0.000771735)
session.viewports['Viewport: 3'].view.setValues(width=0.0198483, 
    height=0.081957, viewOffsetX=0.00118399, viewOffsetY=-0.000482842)
session.viewports['Viewport: 4'].view.setValues(width=0.0194028, 
    height=0.0819723, viewOffsetX=0.0016075, viewOffsetY=-0.000340725)
session.viewports['Viewport: 10'].view.setValues(width=0.0211293, 
    height=0.0819418, viewOffsetX=-9.39462e-05, viewOffsetY=-0.000625398)
session.viewports['Viewport: 11'].view.setValues(width=0.0193951, 
    height=0.0819398, viewOffsetX=0.00160686, viewOffsetY=-0.000340588)
session.viewports['Viewport: 12'].view.setValues(width=0.0204662, 
    height=0.0819309, viewOffsetX=0.00061797, viewOffsetY=-0.000768414)
session.viewports['Viewport: 1'].view.setValues(width=0.0201806, 
    height=0.0772462, viewOffsetX=-0.000421233, viewOffsetY=-0.000639707)
session.viewports['Viewport: 2'].view.setValues(width=0.0185885, 
    height=0.0775806, viewOffsetX=0.00149588, viewOffsetY=-0.000790843)
session.viewports['Viewport: 3'].view.setValues(width=0.0187134, 
    height=0.0772706, viewOffsetX=0.00119171, viewOffsetY=-0.000492621)
session.viewports['Viewport: 4'].view.setValues(width=0.0182932, 
    height=0.0772847, viewOffsetX=0.001629, viewOffsetY=-0.000345905)
session.viewports['Viewport: 10'].view.setValues(width=0.0199212, 
    height=0.0772566, viewOffsetX=-0.000127691, viewOffsetY=-0.000639793)
session.viewports['Viewport: 11'].view.setValues(width=0.018286, 
    height=0.077254, viewOffsetX=0.00162835, viewOffsetY=-0.000345766)
session.viewports['Viewport: 12'].view.setValues(width=0.0192961, 
    height=0.0772467, viewOffsetX=0.000607261, viewOffsetY=-0.00078744)
session.viewports['Viewport: 1'].view.setValues(width=0.0190235, 
    height=0.0728171, viewOffsetX=-0.00046185, viewOffsetY=-0.000653316)
session.viewports['Viewport: 2'].view.setValues(width=0.0175227, 
    height=0.0731326, viewOffsetX=0.00151202, viewOffsetY=-0.00080891)
session.viewports['Viewport: 3'].view.setValues(width=0.0176403, 
    height=0.0728397, viewOffsetX=0.00119902, viewOffsetY=-0.000501868)
session.viewports['Viewport: 4'].view.setValues(width=0.0172442, 
    height=0.0728527, viewOffsetX=0.00164932, viewOffsetY=-0.000350802)
session.viewports['Viewport: 10'].view.setValues(width=0.018779, 
    height=0.0728269, viewOffsetX=-0.000159596, viewOffsetY=-0.000653403)
session.viewports['Viewport: 11'].view.setValues(width=0.0172373, 
    height=0.0728238, viewOffsetX=0.00164866, viewOffsetY=-0.000350661)
session.viewports['Viewport: 12'].view.setValues(width=0.0181898, 
    height=0.0728179, viewOffsetX=0.000597136, viewOffsetY=-0.00080543)
session.viewports[session.currentViewportName].odbDisplay.setFrame(
    step='step2_insitu', frame=103)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports[session.currentViewportName].odbDisplay.setFrame(
    step='step2_insitu', frame=103)
session.viewports['Viewport: 2'].makeCurrent()
session.viewports['Viewport: 11'].makeCurrent()
session.viewports[session.currentViewportName].odbDisplay.setFrame(
    step='step2_insitu', frame=103)
session.viewports[session.currentViewportName].odbDisplay.setFrame(
    step='step2_insitu', frame=103)
session.viewports['Viewport: 12'].makeCurrent()
session.viewports[session.currentViewportName].odbDisplay.setFrame(
    step='step2_insitu', frame=102)
session.viewports['Viewport: 3'].makeCurrent()
session.viewports[session.currentViewportName].odbDisplay.setFrame(
    step='step2_insitu', frame=102)
session.viewports['Viewport: 4'].makeCurrent()
session.viewports[session.currentViewportName].odbDisplay.setFrame(
    step='step2_insitu', frame=103)
session.viewports['Viewport: 4'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM10', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 4'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM2', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 4'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM9', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 4'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM6', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 4'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM7', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 4'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM6', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 4'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM7', outputPosition=INTEGRATION_POINT, )
