# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 2023.HF4 replay file
# Internal Version: 2023_07_21-20.45.57 RELr425 183702
# Run by nguyenb5 on Wed Nov 13 17:11:27 2024
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=103.244789123535, 
    height=113.476852416992)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()
from caeModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
openMdb('TDS_charging.cae')
#: The model database "C:\LocalUserData\User-data\nguyenb5\Abaqus-TDS-Hydrogen-Bayesian-Optimization\targets\CP1000_RD_20C\TDS_charging.cae" has been opened.
session.viewports['Viewport: 1'].setValues(displayedObject=None)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=ON)
p = mdb.models['CP1000'].parts['Part-1']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=ON, 
    engineeringFeatures=ON)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=OFF)
del mdb.models['CP1000'].materials['Material-1'].userOutputVariables
mdb.models['CP1000'].materials['Material-1'].depvar.setValues(n=16)
a = mdb.models['CP1000'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Step-1')
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON, 
    predefinedFields=ON, connectors=ON, optimizationTasks=OFF, 
    geometricRestrictions=OFF, stopConditions=OFF)
del mdb.models['CP1000'].loads['Load-1']
mdb.save()
#: The model database has been saved to "C:\LocalUserData\User-data\nguyenb5\Abaqus-TDS-Hydrogen-Bayesian-Optimization\targets\CP1000_RD_20C\TDS_charging.cae".
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=OFF, bcs=OFF, 
    predefinedFields=OFF, connectors=OFF)
mdb.jobs['CP1000_diffusion'].writeInput(consistencyChecking=OFF)
#: The job input file has been written to "CP1000_diffusion.inp".
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0273867, 
    farPlane=0.0472006, width=0.0255676, height=0.0103323, 
    viewOffsetX=0.00207407, viewOffsetY=0.000107045)
p1 = mdb.models['CP1000'].parts['Part-1']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
mdb.models['CP1000'].materials['Material-1'].depvar.setValues(n=18)
mdb.save()
#: The model database has been saved to "C:\LocalUserData\User-data\nguyenb5\Abaqus-TDS-Hydrogen-Bayesian-Optimization\targets\CP1000_RD_20C\TDS_charging.cae".
a = mdb.models['CP1000'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
mdb.save()
#: The model database has been saved to "C:\LocalUserData\User-data\nguyenb5\Abaqus-TDS-Hydrogen-Bayesian-Optimization\targets\CP1000_RD_20C\TDS_charging.cae".
mdb.save()
#: The model database has been saved to "C:\LocalUserData\User-data\nguyenb5\Abaqus-TDS-Hydrogen-Bayesian-Optimization\targets\CP1000_RD_20C\TDS_charging.cae".
session.viewports['Viewport: 1'].assemblyDisplay.setValues(
    adaptiveMeshConstraints=ON)
mdb.models['CP1000'].TimePoint(name='TDS_measurement_time', points=((120.0, ), 
    (600.0, ), (1800.0, ), (7200.0, )))
mdb.models['CP1000'].fieldOutputRequests['F-Output-1'].setValues(variables=(
    'SDV', ), timePoint='TDS_measurement_time')
mdb.save()
#: The model database has been saved to "C:\LocalUserData\User-data\nguyenb5\Abaqus-TDS-Hydrogen-Bayesian-Optimization\targets\CP1000_RD_20C\TDS_charging.cae".
p1 = mdb.models['CP1000'].parts['Part-1']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
a = mdb.models['CP1000'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON, 
    predefinedFields=ON, connectors=ON, adaptiveMeshConstraints=OFF)
mdb.save()
#: The model database has been saved to "C:\LocalUserData\User-data\nguyenb5\Abaqus-TDS-Hydrogen-Bayesian-Optimization\targets\CP1000_RD_20C\TDS_charging.cae".
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=OFF, bcs=OFF, 
    predefinedFields=OFF, connectors=OFF)
mdb.jobs['CP1000_diffusion'].writeInput(consistencyChecking=OFF)
#: The job input file has been written to "CP1000_diffusion.inp".
session.viewports['Viewport: 1'].assemblyDisplay.setValues(
    adaptiveMeshConstraints=ON)
mdb.models['CP1000'].steps['Step-1'].setValues(initialInc=120.0)
mdb.save()
#: The model database has been saved to "C:\LocalUserData\User-data\nguyenb5\Abaqus-TDS-Hydrogen-Bayesian-Optimization\targets\CP1000_RD_20C\TDS_charging.cae".
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0270237, 
    farPlane=0.0475636, width=0.0252288, height=0.0101954, 
    viewOffsetX=0.00204861, viewOffsetY=7.9568e-05)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(
    adaptiveMeshConstraints=OFF)
mdb.jobs['CP1000_diffusion'].writeInput(consistencyChecking=OFF)
#: The job input file has been written to "CP1000_diffusion.inp".
session.viewports['Viewport: 1'].assemblyDisplay.setValues(
    adaptiveMeshConstraints=ON)
mdb.save()
#: The model database has been saved to "C:\LocalUserData\User-data\nguyenb5\Abaqus-TDS-Hydrogen-Bayesian-Optimization\targets\CP1000_RD_20C\TDS_charging.cae".
session.viewports['Viewport: 1'].assemblyDisplay.setValues(
    adaptiveMeshConstraints=OFF)
mdb.jobs['CP1000_diffusion'].writeInput(consistencyChecking=OFF)
#: The job input file has been written to "CP1000_diffusion.inp".
session.viewports['Viewport: 1'].setValues(displayedObject=None)
mdb.save()
#: The model database has been saved to "C:\LocalUserData\User-data\nguyenb5\Abaqus-TDS-Hydrogen-Bayesian-Optimization\targets\CP1000_RD_20C\TDS_charging.cae".
