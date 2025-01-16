# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 2023.HF4 replay file
# Internal Version: 2023_07_21-20.45.57 RELr425 183702
# Run by nguyenb5 on Tue Nov 26 21:16:36 2024
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=74.578125, 
    height=116.380783081055)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()
from caeModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
openMdb('TDS_charging_CP1000_RD_20C.cae')
#: The model database "C:\LocalUserData\User-data\nguyenb5\Abaqus-TDS-Hydrogen-Bayesian-Optimization\templates\TDS_charging_CP1000_RD_20C.cae" has been opened.
session.viewports['Viewport: 1'].setValues(displayedObject=None)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=ON)
p = mdb.models['CP1000'].parts['Part-1']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0278949, 
    farPlane=0.0466924, width=0.0175144, height=0.00821664, 
    viewOffsetX=0.00114565, viewOffsetY=6.72159e-05)
session.viewports['Viewport: 1'].partDisplay.setValues(mesh=ON)
session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
    meshTechnique=ON)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=OFF)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0280569, 
    farPlane=0.0465304, width=0.0165592, height=0.00963318, 
    viewOffsetX=0.00215531, viewOffsetY=0.000365823)
session.graphicsOptions.setValues(backgroundStyle=SOLID, 
    backgroundColor='#FFFFFF')
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0287711, 
    farPlane=0.0436632, width=0.0169807, height=0.00987837, cameraPosition=(
    0.0230252, 0.0155781, 0.0303103), cameraUpVector=(0.158561, 0.656106, 
    -0.737823), cameraTarget=(0.00744917, 0.0029354, -0.00112804), 
    viewOffsetX=0.00221016, viewOffsetY=0.000375134)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0296341, 
    farPlane=0.0392689, width=0.0174901, height=0.0101747, cameraPosition=(
    0.00379824, 0.0117333, 0.0333734), cameraUpVector=(0.424611, 0.762357, 
    -0.488383), cameraTarget=(0.00782276, 0.0027536, -0.00259863), 
    viewOffsetX=0.00227646, viewOffsetY=0.000386387)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0209515, 
    farPlane=0.0434546, width=0.0123656, height=0.00719357, cameraPosition=(
    -0.0229947, 0.00888321, 0.00849364), cameraUpVector=(0.493338, 0.863864, 
    -0.101771), cameraTarget=(0.0118798, 0.00202663, -0.00280139), 
    viewOffsetX=0.00160947, viewOffsetY=0.000273178)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0231182, 
    farPlane=0.0423425, width=0.0136444, height=0.00793751, cameraPosition=(
    -0.0180705, 0.0182942, 0.0129357), cameraUpVector=(0.570124, 0.645182, 
    -0.508625), cameraTarget=(0.0102915, -4.44555e-06, -0.00292548), 
    viewOffsetX=0.00177592, viewOffsetY=0.000301429)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0231475, 
    farPlane=0.0423518, width=0.0136617, height=0.00794758, cameraPosition=(
    -0.0173675, 0.0173216, 0.0154045), cameraUpVector=(0.531287, 0.678101, 
    -0.507852), cameraTarget=(0.0100663, 0.000228385, -0.00319742), 
    viewOffsetX=0.00177817, viewOffsetY=0.000301811)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0234665, 
    farPlane=0.042054, width=0.01385, height=0.00805711, cameraPosition=(
    -0.0164464, 0.0158912, 0.0181025), cameraUpVector=(0.514114, 0.724166, 
    -0.459641), cameraTarget=(0.00982964, 0.000664301, -0.00354322), 
    viewOffsetX=0.00180268, viewOffsetY=0.00030597)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0233051, 
    farPlane=0.0422155, width=0.0156304, height=0.00909285, 
    viewOffsetX=0.00238417, viewOffsetY=-0.000908671)
a = mdb.models['CP1000'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON, 
    predefinedFields=ON, connectors=ON, optimizationTasks=OFF, 
    geometricRestrictions=OFF, stopConditions=OFF)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0284589, 
    farPlane=0.0461284, width=0.014468, height=0.00838279, cameraPosition=(
    0.0297477, 0.023124, 0.0217228), cameraUpVector=(-0.283064, 0.514889, 
    -0.809175))
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0308895, 
    farPlane=0.0424469, width=0.0157037, height=0.00909874, cameraPosition=(
    0.000174289, 0.0216981, 0.0305543), cameraUpVector=(0.192447, 0.614315, 
    -0.765233), cameraTarget=(0.0082162, 0.0015925, 0.000191291))
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0259458, 
    farPlane=0.0465898, width=0.0131904, height=0.00764255, cameraPosition=(
    -0.021709, 0.0142592, 0.018167), cameraUpVector=(0.470595, 0.766027, 
    -0.437885), cameraTarget=(0.00858949, 0.00171939, 0.000402599))
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0263755, 
    farPlane=0.0461601, width=0.0134089, height=0.00776914, cameraPosition=(
    -0.021709, 0.0142592, 0.018167), cameraUpVector=(0.545365, 0.775829, 
    -0.317279), cameraTarget=(0.00858949, 0.00171939, 0.000402599))
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0269401, 
    farPlane=0.0456976, width=0.0136959, height=0.00793545, cameraPosition=(
    -0.0178643, 0.0197684, 0.0194794), cameraUpVector=(0.676918, 0.653124, 
    -0.339429), cameraTarget=(0.00848074, 0.00156356, 0.000365476))
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0274068, 
    farPlane=0.0453371, width=0.0139332, height=0.00807292, cameraPosition=(
    -0.0164014, 0.0161637, 0.0239787), cameraUpVector=(0.555873, 0.733891, 
    -0.390396), cameraTarget=(0.00844147, 0.00166031, 0.00024471))
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0288699, 
    farPlane=0.0440873, width=0.014677, height=0.0085039, cameraPosition=(
    -0.0108499, 0.0118795, 0.0304428), cameraUpVector=(0.424534, 0.807051, 
    -0.410414), cameraTarget=(0.00830078, 0.00176888, 8.08968e-05))
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0266708, 
    farPlane=0.045988, width=0.013559, height=0.00785612, cameraPosition=(
    -0.0189505, 0.0161272, 0.0210134), cameraUpVector=(0.549949, 0.739705, 
    -0.387805), cameraTarget=(0.00848178, 0.00167397, 0.000291585))
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.026873, 
    farPlane=0.0457649, width=0.0136618, height=0.00791567, cameraPosition=(
    -0.0178519, 0.0210354, 0.0182252), cameraUpVector=(0.637663, 0.629531, 
    -0.443933), cameraTarget=(0.00845262, 0.00154369, 0.000365592))
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0269615, 
    farPlane=0.0457138, width=0.0137068, height=0.00794173, cameraPosition=(
    -0.0181201, 0.0175686, 0.0210387), cameraUpVector=(0.557008, 0.710042, 
    -0.430793), cameraTarget=(0.00845982, 0.00163673, 0.000290081))
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0266898, 
    farPlane=0.0459856, width=0.0144347, height=0.00836354, 
    viewOffsetX=5.84386e-05, viewOffsetY=-0.000273715)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.026606, 
    farPlane=0.0459909, width=0.0143894, height=0.00833728, cameraPosition=(
    -0.0187462, 0.0188258, 0.0190951), cameraUpVector=(0.57246, 0.680839, 
    -0.456889), cameraTarget=(0.00848239, 0.00159234, 0.000321944), 
    viewOffsetX=5.82551e-05, viewOffsetY=-0.000272856)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Step-1')
#: 
#: Point 1: 0., 4.E-03, 0.  Point 2: 0., 4.E-03, 1.E-03
#:    Distance: 1.E-03  Components: 0., 0., 1.E-03
#: 
#: Point 1: 0., 4.E-03, 1.E-03  Point 2: 0., 0., 1.E-03
#:    Distance: 4.E-03  Components: 0., -4.E-03, 0.
#: 
#: Point 1: 15.E-03, 4.E-03, 1.E-03  Point 2: 0., 4.E-03, 1.E-03
#:    Distance: 15.E-03  Components: -15.E-03, 0., 0.
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0257311, 
    farPlane=0.0468657, width=0.018557, height=0.00712458, 
    viewOffsetX=0.000717784, viewOffsetY=-0.000341415)
session.viewports['Viewport: 1'].setValues(displayedObject=None)
o1 = session.openOdb(
    name='C:/LocalUserData/User-data/nguyenb5/CP1000 plastic (UMAT UMATHT)/CP1000_dense_CSC/CP1000_SH115_dense_newBC/SH115_combined_processed.odb')
session.viewports['Viewport: 1'].setValues(displayedObject=o1)
#: Model: C:/LocalUserData/User-data/nguyenb5/CP1000 plastic (UMAT UMATHT)/CP1000_dense_CSC/CP1000_SH115_dense_newBC/SH115_combined_processed.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       9
#: Number of Node Sets:          8
#: Number of Steps:              2
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.462208, 
    farPlane=0.621791, width=0.0128937, height=0.00802641, 
    viewOffsetX=-0.00189664, viewOffsetY=-0.00869275)
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=102 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=101 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=100 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=99 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=98 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=97 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=96 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=95 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=94 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=93 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=92 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=91 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=90 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=89 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=88 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=87 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=86 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=85 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=84 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=83 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=82 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=81 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=80 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=0 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=1 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=2 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=3 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=4 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=5 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=6 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=7 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=8 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=9 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=10 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=11 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=12 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=13 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=14 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=15 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=16 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=17 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=18 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=19 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=20 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=21 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=22 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=23 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=24 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=25 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=26 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=27 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=28 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=29 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=30 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=31 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=32 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=33 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=34 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=35 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=36 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=37 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=38 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=39 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=40 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=39 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=38 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=37 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=36 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=35 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=34 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=33 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=103 )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM1', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.463949, 
    farPlane=0.62005, width=0.00375462, height=0.00233728, 
    viewOffsetX=-0.00057744, viewOffsetY=-0.00955365)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.53473, 
    farPlane=0.547739, width=0.00432743, height=0.00269386, cameraPosition=(
    0.391066, 0.0110202, -0.374168), cameraUpVector=(-0.526279, 0.847093, 
    -0.0739184), cameraTarget=(-0.00477827, 0.0110848, -0.00279163), 
    viewOffsetX=-0.000665535, viewOffsetY=-0.0110112)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.497677, 
    farPlane=0.584791, width=0.00402757, height=0.0025072, cameraPosition=(
    0.114199, -0.164043, -0.502516), cameraUpVector=(0.209261, 0.975736, 
    0.0644169), cameraTarget=(0.00245016, 0.012272, -0.00147854), 
    viewOffsetX=-0.000619418, viewOffsetY=-0.0102482)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.506412, 
    farPlane=0.57638, width=0.00409826, height=0.00255121, cameraPosition=(
    0.0897785, 0.152684, -0.511819), cameraUpVector=(0.0245601, 0.816117, 
    0.577364), cameraTarget=(0.000136025, 0.0117204, 0.0046176), 
    viewOffsetX=-0.00063029, viewOffsetY=-0.0104281)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.505874, 
    farPlane=0.576918, width=0.00763193, height=0.00475094, 
    viewOffsetX=-0.000396012, viewOffsetY=-0.00997804)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.486212, 
    farPlane=0.596069, width=0.00733529, height=0.00456627, cameraPosition=(
    -0.210248, -0.204862, -0.45386), cameraUpVector=(0.15975, 0.976724, 
    -0.143143), cameraTarget=(0.000371912, 0.011996, -0.00305497), 
    viewOffsetX=-0.000380619, viewOffsetY=-0.00959021)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.529131, 
    farPlane=0.553312, width=0.00798279, height=0.00496934, cameraPosition=(
    0.0570865, 0.0539824, -0.535465), cameraUpVector=(-0.123137, 0.910959, 
    0.393688), cameraTarget=(-0.00180157, 0.012147, 0.00249006), 
    viewOffsetX=-0.000414217, viewOffsetY=-0.0104368)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.450164, 
    farPlane=0.630912, width=0.00679145, height=0.00422772, cameraPosition=(
    0.350762, -0.350899, 0.21362), cameraUpVector=(0.400249, 0.914759, 
    0.0549141), cameraTarget=(0.00588692, 0.0105273, 0.00137984), 
    viewOffsetX=-0.0003524, viewOffsetY=-0.00887923)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.493955, 
    farPlane=0.587487, width=0.00745211, height=0.00463899, cameraPosition=(
    0.176286, -0.174865, 0.480416), cameraUpVector=(0.500774, 0.833841, 
    -0.232238), cameraTarget=(0.00665104, 0.00993094, -0.000924764), 
    viewOffsetX=-0.000386681, viewOffsetY=-0.00974299)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.520756, 
    farPlane=0.560693, width=0.00785644, height=0.00489069, cameraPosition=(
    0.489456, 0.059111, 0.223401), cameraUpVector=(-0.114376, 0.708977, 
    -0.695895), cameraTarget=(0.000564056, 0.00922914, -0.00705706), 
    viewOffsetX=-0.000407661, viewOffsetY=-0.0102716)
session.viewports['Viewport: 1'].view.setValues(session.views['Iso'])
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.438439, 
    farPlane=0.656761, width=0.175107, height=0.109005, viewOffsetX=0.00124207, 
    viewOffsetY=-0.0052943)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.451673, 
    farPlane=0.644698, width=0.180392, height=0.112296, cameraPosition=(
    0.468605, 0.27155, 0.0904459), cameraUpVector=(-0.692409, 0.661569, 
    -0.287917), cameraTarget=(-0.00423593, 0.0117588, -0.00334425), 
    viewOffsetX=0.00127956, viewOffsetY=-0.0054541)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.458326, 
    farPlane=0.642761, width=0.183049, height=0.11395, cameraPosition=(
    0.413186, 0.252028, -0.263744), cameraUpVector=(-0.587546, 0.700453, 
    0.405161), cameraTarget=(-0.00254567, 0.0116723, -0.000566238), 
    viewOffsetX=0.00129841, viewOffsetY=-0.00553444)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.471773, 
    farPlane=0.633994, width=0.18842, height=0.117293, cameraPosition=(
    0.165252, 0.21676, -0.481424), cameraUpVector=(-0.283349, 0.747165, 
    0.601214), cameraTarget=(-0.00219752, 0.0118074, -0.00202686), 
    viewOffsetX=0.00133651, viewOffsetY=-0.00569682)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.471216, 
    farPlane=0.638599, width=0.188198, height=0.117155, cameraPosition=(
    -0.091025, 0.229015, -0.497458), cameraUpVector=(-0.0471837, 0.719837, 
    0.692537), cameraTarget=(-0.00400012, 0.0119303, -0.00231468), 
    viewOffsetX=0.00133493, viewOffsetY=-0.0056901)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.484518, 
    farPlane=0.625296, width=0.104228, height=0.0648828, 
    viewOffsetX=-0.000426998, viewOffsetY=-0.00767218)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.522003, 
    farPlane=0.587542, width=0.112292, height=0.0699025, cameraPosition=(
    -0.0616968, 0.0707273, -0.546695), cameraUpVector=(-0.0764643, 0.892424, 
    0.444672), cameraTarget=(-0.00372335, 0.0105701, -0.00550535), 
    viewOffsetX=-0.000460033, viewOffsetY=-0.00826575)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.539307, 
    farPlane=0.570239, width=0.00321885, height=0.00200376, 
    viewOffsetX=-0.00296332, viewOffsetY=-0.00784392)
session.viewports[session.currentViewportName].odbDisplay.setFrame(
    step='step2_insitu', frame=64)
session.viewports[session.currentViewportName].odbDisplay.setFrame(
    step='step2_insitu', frame=103)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='SDV_AR2_SIG22', outputPosition=INTEGRATION_POINT, )
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
    step='step2_insitu', frame=67)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.539153, 
    farPlane=0.570468, width=0.00466455, height=0.00290372, 
    viewOffsetX=-0.00305903, viewOffsetY=-0.00793642)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM1', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.539523, 
    farPlane=0.570098, width=0.00208819, height=0.00129991, 
    viewOffsetX=-0.00250006, viewOffsetY=-0.0081442)
session.viewports[session.currentViewportName].odbDisplay.setFrame(
    step='step2_insitu', frame=97)
session.viewports[session.currentViewportName].odbDisplay.setFrame(
    step='step2_insitu', frame=98)
session.viewports[session.currentViewportName].odbDisplay.setFrame(
    step='step2_insitu', frame=103)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='SDV_AR4_SIG12', outputPosition=INTEGRATION_POINT, )
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
    step='step2_insitu', frame=100)
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=99 )
#: Warning: The selected Primary Variable is not available in the current step/frame.
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=98 )
#: Warning: The selected Primary Variable is not available in the current step/frame.
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=97 )
#: Warning: The selected Primary Variable is not available in the current step/frame.
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=96 )
#: Warning: The selected Primary Variable is not available in the current step/frame.
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=95 )
#: Warning: The selected Primary Variable is not available in the current step/frame.
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=94 )
#: Warning: The selected Primary Variable is not available in the current step/frame.
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=93 )
#: Warning: The selected Primary Variable is not available in the current step/frame.
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=92 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=91 )
#: Warning: The selected Primary Variable is not available in the current step/frame.
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=1, frame=92 )
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.539442, 
    farPlane=0.570131, width=0.00251375, height=0.00156483, 
    viewOffsetX=-0.00248525, viewOffsetY=-0.00814136)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM1', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.539181, 
    farPlane=0.570392, width=0.00365694, height=0.00227647, 
    viewOffsetX=-0.00262878, viewOffsetY=-0.0078273)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.536955, 
    farPlane=0.573081, width=0.00364184, height=0.00226707, cameraPosition=(
    0.19848, 0.0720293, -0.5133), cameraUpVector=(-0.445006, 0.847583, 
    0.289089), cameraTarget=(-0.00180946, 0.00970271, -0.0074682), 
    viewOffsetX=-0.00261793, viewOffsetY=-0.00779498)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.529087, 
    farPlane=0.581851, width=0.00358847, height=0.00223385, cameraPosition=(
    0.522869, 0.0874175, -0.166944), cameraUpVector=(-0.591934, 0.672514, 
    -0.444228), cameraTarget=(0.00485922, 0.00771036, -0.00826486), 
    viewOffsetX=-0.00257957, viewOffsetY=-0.00768076)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.496421, 
    farPlane=0.616024, width=0.00336692, height=0.00209593, cameraPosition=(
    0.331882, 0.24451, 0.374631), cameraUpVector=(0.216911, 0.366394, 
    -0.904823), cameraTarget=(0.0101521, 0.00790982, -3.81148e-05), 
    viewOffsetX=-0.00242031, viewOffsetY=-0.00720655)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.488232, 
    farPlane=0.624821, width=0.00331138, height=0.00206135, cameraPosition=(
    0.0605035, 0.300768, 0.465353), cameraUpVector=(0.436991, 0.489963, 
    -0.754305), cameraTarget=(0.0069084, 0.0107717, 0.00394616), 
    viewOffsetX=-0.00238038, viewOffsetY=-0.00708766)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.487757, 
    farPlane=0.625296, width=0.00577345, height=0.00359402, 
    viewOffsetX=-0.0026098, viewOffsetY=-0.0069717)
session.viewports['Viewport: 1'].view.setValues(session.views['Iso'])
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.468718, 
    farPlane=0.625495, width=0.00625359, height=0.00389291, 
    viewOffsetX=-0.000335512, viewOffsetY=-0.00970335)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.465797, 
    farPlane=0.628331, width=0.00621462, height=0.00386865, cameraPosition=(
    0.435923, 0.331957, 0.00601654), cameraUpVector=(-0.822919, 0.567077, 
    0.0350316), cameraTarget=(-0.00710978, 0.0110005, 0.000174032), 
    viewOffsetX=-0.000333421, viewOffsetY=-0.00964289)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.482059, 
    farPlane=0.613055, width=0.00643158, height=0.00400371, cameraPosition=(
    0.076099, 0.285261, -0.461665), cameraUpVector=(-0.291868, 0.636401, 
    0.714008), cameraTarget=(-0.00372876, 0.0114165, 0.00519899), 
    viewOffsetX=-0.000345061, viewOffsetY=-0.00997953)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.540171, 
    farPlane=0.554317, width=0.0072069, height=0.00448636, cameraPosition=(
    0.0395962, -0.00939189, -0.545576), cameraUpVector=(-0.148886, 0.946992, 
    0.284675), cameraTarget=(-0.00233161, 0.0127464, -0.000528369), 
    viewOffsetX=-0.000386658, viewOffsetY=-0.0111826)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.537297, 
    farPlane=0.55719, width=0.0247102, height=0.0153823, 
    viewOffsetX=0.000127627, viewOffsetY=-0.0104809)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.494398, 
    farPlane=0.598448, width=0.0227372, height=0.0141541, cameraPosition=(
    0.501511, 0.186588, 0.114091), cameraUpVector=(-0.407503, 0.600883, 
    -0.687663), cameraTarget=(-0.00213599, 0.0103037, -0.00669196), 
    viewOffsetX=0.000117437, viewOffsetY=-0.00964404)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.48293, 
    farPlane=0.610712, width=0.0222098, height=0.0138258, cameraPosition=(
    0.280278, 0.253524, 0.396431), cameraUpVector=(-0.126112, 0.644842, 
    -0.75384), cameraTarget=(-0.000275403, 0.0111391, -0.00589334), 
    viewOffsetX=0.000114713, viewOffsetY=-0.00942034)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.485214, 
    farPlane=0.60843, width=0.00882089, height=0.00549108, 
    viewOffsetX=-0.000311464, viewOffsetY=-0.00928865)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.498746, 
    farPlane=0.595131, width=0.0090669, height=0.00564422, cameraPosition=(
    0.143195, 0.203817, 0.487811), cameraUpVector=(0.288527, 0.647039, 
    -0.705757), cameraTarget=(0.0034577, 0.0108008, -0.00467619), 
    viewOffsetX=-0.00032015, viewOffsetY=-0.00954771)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.526313, 
    farPlane=0.567762, width=0.00956806, height=0.00595619, cameraPosition=(
    -0.0680057, 0.0891926, 0.535958), cameraUpVector=(0.484775, 0.783937, 
    -0.387861), cameraTarget=(0.00426496, 0.0115815, -0.000772242), 
    viewOffsetX=-0.000337846, viewOffsetY=-0.0100754)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.527125, 
    farPlane=0.566951, width=0.00459806, height=0.00286233, 
    viewOffsetX=0.000601407, viewOffsetY=-0.0107044)
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
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='UVARM1', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.463819, 
    farPlane=0.620874, width=0.00791923, height=0.00492978, 
    viewOffsetX=-0.00103155, viewOffsetY=-0.0100056)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.505241, 
    farPlane=0.577278, width=0.00862647, height=0.00537005, cameraPosition=(
    0.275595, -0.120606, -0.449655), cameraUpVector=(-0.264291, 0.963403, 
    -0.0447676), cameraTarget=(-0.00320319, 0.0114642, -0.00306534), 
    viewOffsetX=-0.00112367, viewOffsetY=-0.0108992)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.494827, 
    farPlane=0.587977, width=0.00844866, height=0.00525936, cameraPosition=(
    0.0480918, -0.178972, -0.508309), cameraUpVector=(0.221194, 0.975178, 
    0.0100512), cameraTarget=(0.00156445, 0.0118992, -0.00232779), 
    viewOffsetX=-0.00110051, viewOffsetY=-0.0106745)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.527276, 
    farPlane=0.555222, width=0.00900269, height=0.00560425, cameraPosition=(
    0.233239, -0.027308, -0.487496), cameraUpVector=(-0.325732, 0.935321, 
    0.138108), cameraTarget=(-0.00414612, 0.0116145, -0.000929668), 
    viewOffsetX=-0.00117268, viewOffsetY=-0.0113745)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.529394, 
    farPlane=0.553195, width=0.00903886, height=0.00562676, cameraPosition=(
    0.11501, -0.0247743, -0.528195), cameraUpVector=(-0.241318, 0.944523, 
    0.222805), cameraTarget=(-0.00376444, 0.0117977, 0.000168812), 
    viewOffsetX=-0.00117739, viewOffsetY=-0.0114202)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.53581, 
    farPlane=0.547003, width=0.00914841, height=0.00569496, cameraPosition=(
    -0.233945, 0.0096966, -0.488056), cameraUpVector=(0.178818, 0.943461, 
    0.279116), cameraTarget=(-0.000206254, 0.0122403, 0.00181432), 
    viewOffsetX=-0.00119166, viewOffsetY=-0.0115586)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.536841, 
    farPlane=0.545972, width=0.00320151, height=0.00199296, 
    viewOffsetX=-0.000160052, viewOffsetY=-0.011473)
o7 = session.odbs['C:/LocalUserData/User-data/nguyenb5/CP1000 plastic (UMAT UMATHT)/CP1000_dense_CSC/CP1000_SH115_dense_newBC/SH115_combined_processed.odb']
session.viewports['Viewport: 1'].setValues(displayedObject=o7)
