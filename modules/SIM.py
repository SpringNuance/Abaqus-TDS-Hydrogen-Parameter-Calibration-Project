import pandas as pd
import numpy as np
import subprocess
import os
import matplotlib.pyplot as mp
from modules.calculation import *
from modules.IO import *
from modules.run_sim import *
from optimizers.optimize import *
import sys
import shutil
import random
import time
import sobol_seq

class SIM():
    def __init__(self, info):
        self.info = info
   
    def latin_hypercube_sampling(self):
        paramConfig = self.info["paramConfig"]
        numberOfInitialSims = self.info["numberOfInitialSims"]
        linspaceValues = {}
        for param in paramConfig:
            linspaceValues[param] = np.linspace(
                start=paramConfig[param]["lowerBound"] * paramConfig[param]["exponent"], 
                stop=paramConfig[param]["upperBound"] * paramConfig[param]["exponent"], 
                num = self.info["initialSimsSpacing"])
            linspaceValues[param] = linspaceValues[param].tolist()   
        points = []
        for _ in range(numberOfInitialSims):
            while True:
                candidateParam = {}
                for param in linspaceValues:
                    random.shuffle(linspaceValues[param])
                    candidateParam[param] = linspaceValues[param].pop()
                if candidateParam not in points:
                    break
            points.append(candidateParam)

        return points
    
    def sobol_sequence_sampling(self):
        paramConfig = self.info["paramConfig"]
        numberOfInitialSims = self.info["numberOfInitialSims"]
        num_params = len(paramConfig)

        # Generate Sobol sequence samples
        sobol_samples = sobol_seq.i4_sobol_generate(num_params, numberOfInitialSims)

        # Scale samples to parameter ranges
        points = []
        for sample in sobol_samples:
            scaled_sample = {}
            for i, param in enumerate(paramConfig):
                lower_bound = paramConfig[param]["lowerBound"] * paramConfig[param]["exponent"]
                upper_bound = paramConfig[param]["upperBound"] * paramConfig[param]["exponent"]
                # Scale the Sobol sample for this parameter
                scaled_sample[param] = lower_bound + (upper_bound - lower_bound) * sample[i]
            points.append(scaled_sample)

        # print(points)
        # time.sleep(180)
        return points

    def run_initial_simulations(self, parameters, index):
        self.preprocess_simulations_initial(parameters, index)
        self.submit_array_jobs_initial(index)
        self.postprocess_results_initial(parameters, index)
        deleteSimOutputs = self.info['deleteSimOutputs']
        if deleteSimOutputs == "yes":
            self.delete_sim_outputs_initial(index)

    def preprocess_simulations_initial(self, parameters, index):
        fractureModel = self.info['fractureModel']
        resultPath = self.info['resultPath']
        simPath = self.info['simPath']
        templatePath = self.info['templatePath'] 
        geometries = self.info['geometries']
        timeSteps = self.info['timeSteps']
        numFrames = self.info['numFrames']
        maxTargetDisplacements = self.info['maxTargetDisplacements']

        for geometry in geometries:
            if os.path.exists(f"{simPath}/{geometry}/initial/{index}"):
                shutil.rmtree(f"{simPath}/{geometry}/initial/{index}")
            shutil.copytree(f"{templatePath}/{geometry}", f"{simPath}/{geometry}/initial/{index}")
            #print(geometry)
            if fractureModel == "eMBW":
                replace_eMBW_fracture_params(f"{simPath}/{geometry}/initial/{index}/material.inp", parameters)
            elif fractureModel == "eMBW+Hill48":
                replace_eMBW_Hill48_fracture_params(f"{simPath}/{geometry}/initial/{index}/material.inp", parameters)
            replace_materialName_geometry_inp(f"{simPath}/{geometry}/initial/{index}/geometry.inp", "material.inp")
            replace_maxDisp_geometry_inp(f"{simPath}/{geometry}/initial/{index}/geometry.inp", maxTargetDisplacements[geometry])
            replace_timeStep_geometry_inp(f"{simPath}/{geometry}/initial/{index}/geometry.inp", timeSteps[geometry])
            replace_numFrames_geometry_inp(f"{simPath}/{geometry}/initial/{index}/geometry.inp", numFrames[geometry])
            create_parameter_file(f"{simPath}/{geometry}/initial/{index}", parameters)

    def preprocess_simulations_initial_no_overwrite(self, parameters, index):
        fractureModel = self.info['fractureModel']
        simPath = self.info['simPath']
        templatePath = self.info['templatePath'] 
        geometries = self.info['geometries']
        timeSteps = self.info['timeSteps']
        numFrames = self.info['numFrames']
        maxTargetDisplacements = self.info['maxTargetDisplacements']

        for geometry in geometries:
            if not os.path.exists(f"{simPath}/{geometry}/initial/{index}"):
                shutil.copytree(f"{templatePath}/{geometry}", f"{simPath}/{geometry}/initial/{index}")
                #print(geometry)
                if fractureModel == "eMBW":
                    replace_eMBW_fracture_params(f"{simPath}/{geometry}/initial/{index}/material.inp", parameters)
                elif fractureModel == "eMBW+Hill48":
                    replace_eMBW_Hill48_fracture_params(f"{simPath}/{geometry}/initial/{index}/material.inp", parameters)
                replace_materialName_geometry_inp(f"{simPath}/{geometry}/initial/{index}/geometry.inp", "material.inp")
                replace_maxDisp_geometry_inp(f"{simPath}/{geometry}/initial/{index}/geometry.inp", maxTargetDisplacements[geometry])
                replace_timeStep_geometry_inp(f"{simPath}/{geometry}/initial/{index}/geometry.inp", timeSteps[geometry])
                replace_numFrames_geometry_inp(f"{simPath}/{geometry}/initial/{index}/geometry.inp", numFrames[geometry])
                create_parameter_file(f"{simPath}/{geometry}/initial/{index}", parameters)
            
    def submit_array_jobs_initial(self, index):
        projectPath = self.info['projectPath']
        simPath = self.info['simPath']
        geometries = self.info['geometries']
        logPath = self.info['logPath']
 
        commands = []
        paths = []
        for geometry in geometries:
            commands.append("start /wait cmd /c run.bat")
            paths.append(f"{projectPath}/{simPath}/{geometry}/initial/{index}")
    
        printLog(f"Number of called subprocesses required: {len(geometries)}", logPath)
        #time.sleep(180)
        run_bat_files_parallel(commands, paths)
        printLog("Initial simulations for the parameters have finished", logPath)
    
    def postprocess_results_initial(self, parameters, index):
        numberOfInitialSims = self.info['numberOfInitialSims']
        simPath = self.info['simPath']
        resultPath = self.info['resultPath']
        logPath = self.info['logPath']
        geometries = self.info['geometries']
        # The structure of force-displacement curve: dict of (hardening law params typle) -> {force: forceArray , displacement: displacementArray}

        paramsTuple = tuple(parameters.items())

        for geometry in geometries:
            if os.path.exists(f"{resultPath}/{geometry}/initial/data/{index}"):
                shutil.rmtree(f"{resultPath}/{geometry}/initial/data/{index}")
            os.mkdir(f"{resultPath}/{geometry}/initial/data/{index}")
            shutil.copy(f"{simPath}/{geometry}/initial/{index}/FD_Curve.txt", f"{resultPath}/{geometry}/initial/data/{index}")
            shutil.copy(f"{simPath}/{geometry}/initial/{index}/FD_Curve_Plot.tif", f"{resultPath}/{geometry}/initial/data/{index}")
            shutil.copy(f"{simPath}/{geometry}/initial/{index}/Deformed_Specimen.tif", f"{resultPath}/{geometry}/initial/data/{index}")
            shutil.copy(f"{simPath}/{geometry}/initial/{index}/parameters.xlsx", f"{resultPath}/{geometry}/initial/data/{index}")
            shutil.copy(f"{simPath}/{geometry}/initial/{index}/parameters.csv", f"{resultPath}/{geometry}/initial/data/{index}")
                    
            displacement, force = read_FD_Curve(f"{simPath}/{geometry}/initial/{index}/FD_Curve.txt")
            create_FD_Curve_file(f"{resultPath}/{geometry}/initial/data/{index}", displacement, force)
            
        printLog("Saving successfully simulation results", logPath)
    
    def postprocess_results_initial_no_overwrite(self, parameters, index):
        numberOfInitialSims = self.info['numberOfInitialSims']
        simPath = self.info['simPath']
        resultPath = self.info['resultPath']
        logPath = self.info['logPath']
        geometries = self.info['geometries']
        # The structure of force-displacement curve: dict of (hardening law params typle) -> {force: forceArray , displacement: displacementArray}

        paramsTuple = tuple(parameters.items())

        for geometry in geometries:
            if not os.path.exists(f"{resultPath}/{geometry}/initial/data/{index}"):
                if os.path.exists(f"{simPath}/{geometry}/initial/{index}/FD_Curve.txt"):
                    os.mkdir(f"{resultPath}/{geometry}/initial/data/{index}")
                    shutil.copy(f"{simPath}/{geometry}/initial/{index}/FD_Curve.txt", f"{resultPath}/{geometry}/initial/data/{index}")
                    shutil.copy(f"{simPath}/{geometry}/initial/{index}/FD_Curve_Plot.tif", f"{resultPath}/{geometry}/initial/data/{index}")
                    shutil.copy(f"{simPath}/{geometry}/initial/{index}/Deformed_Specimen.tif", f"{resultPath}/{geometry}/initial/data/{index}")
                    shutil.copy(f"{simPath}/{geometry}/initial/{index}/parameters.xlsx", f"{resultPath}/{geometry}/initial/data/{index}")
                    shutil.copy(f"{simPath}/{geometry}/initial/{index}/parameters.csv", f"{resultPath}/{geometry}/initial/data/{index}")
                            
                    displacement, force = read_FD_Curve(f"{simPath}/{geometry}/initial/{index}/FD_Curve.txt")
                    create_FD_Curve_file(f"{resultPath}/{geometry}/initial/data/{index}", displacement, force)
    
    def delete_sim_outputs_initial(self, index):
        simPath = self.info['simPath']
        geometries = self.info['geometries']
        for geometry in geometries:
            if os.path.exists(f"{simPath}/{geometry}/initial/{index}"):
                shutil.rmtree(f"{simPath}/{geometry}/initial/{index}")
    ############################################################################################################

    # This function is useful for debugging
    def run_dummy_iteration_simulations(self, paramsDict, iterationIndex):
        geom_to_params_FD_Curves = np.load("modules/geom_to_param_FD_Curves.npy", allow_pickle=True).tolist()
        return geom_to_params_FD_Curves
    
    def run_iteration_simulations(self, parameters, index):
        self.preprocess_simulations_iteration(parameters, index)
        self.submit_array_jobs_iteration(index)
        new_geom_to_param_FD_Curves = self.postprocess_results_iteration(parameters, index)
        deleteSimOutputs = self.info['deleteSimOutputs']
        if deleteSimOutputs == "yes":
            self.delete_sim_outputs_iteration(index)
        return new_geom_to_param_FD_Curves

    def preprocess_simulations_iteration(self, parameters, index):
        fractureModel = self.info['fractureModel']
        simPath = self.info['simPath']
        templatePath = self.info['templatePath'] 
        geometries = self.info['geometries']
        timeSteps = self.info['timeSteps']
        numFrames = self.info['numFrames']
        maxTargetDisplacements = self.info['maxTargetDisplacements']

        for geometry in geometries:
            if os.path.exists(f"{simPath}/{geometry}/iteration/{index}"):
                shutil.rmtree(f"{simPath}/{geometry}/iteration/{index}")
            shutil.copytree(f"{templatePath}/{geometry}", f"{simPath}/{geometry}/iteration/{index}")
            if fractureModel == "eMBW":
                replace_eMBW_fracture_params(f"{simPath}/{geometry}/iteration/{index}/material.inp", parameters)
            elif fractureModel == "eMBW+Hill48":
                replace_eMBW_Hill48_fracture_params(f"{simPath}/{geometry}/iteration/{index}/material.inp", parameters)
            replace_materialName_geometry_inp(f"{simPath}/{geometry}/iteration/{index}/geometry.inp", "material.inp")
            replace_maxDisp_geometry_inp(f"{simPath}/{geometry}/iteration/{index}/geometry.inp", maxTargetDisplacements[geometry])
            replace_timeStep_geometry_inp(f"{simPath}/{geometry}/iteration/{index}/geometry.inp", timeSteps[geometry])
            replace_numFrames_geometry_inp(f"{simPath}/{geometry}/iteration/{index}/geometry.inp", numFrames[geometry])
            create_parameter_file(f"{simPath}/{geometry}/iteration/{index}", parameters)

    def submit_array_jobs_iteration(self, index):
        projectPath = self.info['projectPath']
        simPath = self.info['simPath']
        geometries = self.info['geometries']
        logPath = self.info['logPath']
 
        commands = []
        paths = []
        for geometry in geometries:
            commands.append("start /wait cmd /c run.bat")
            paths.append(f"{projectPath}/{simPath}/{geometry}/iteration/{index}")
    
        printLog(f"Number of called subprocesses required: {len(geometries)}", logPath)
        #time.sleep(180)
        run_bat_files_parallel(commands, paths)
        printLog("iteration simulations for the parameters have finished", logPath)
    
    def postprocess_results_iteration(self, parameters, index):
        simPath = self.info['simPath']
        resultPath = self.info['resultPath']
        logPath = self.info['logPath']
        geometries = self.info['geometries']
        # The structure of force-displacement curve: dict of (hardening law params typle) -> {force: forceArray , displacement: displacementArray}

        paramsTuple = tuple(parameters.items())

        new_geom_to_param_FD_Curves = {}

        for geometry in geometries:
            if os.path.exists(f"{resultPath}/{geometry}/iteration/data/{index}"):
                shutil.rmtree(f"{resultPath}/{geometry}/iteration/data/{index}")
            os.mkdir(f"{resultPath}/{geometry}/iteration/data/{index}")
            shutil.copy(f"{simPath}/{geometry}/iteration/{index}/FD_Curve.txt", f"{resultPath}/{geometry}/iteration/data/{index}")
            shutil.copy(f"{simPath}/{geometry}/iteration/{index}/FD_Curve_Plot.tif", f"{resultPath}/{geometry}/iteration/data/{index}")
            shutil.copy(f"{simPath}/{geometry}/iteration/{index}/Deformed_Specimen.tif", f"{resultPath}/{geometry}/iteration/data/{index}")
            shutil.copy(f"{simPath}/{geometry}/iteration/{index}/parameters.xlsx", f"{resultPath}/{geometry}/iteration/data/{index}")
            shutil.copy(f"{simPath}/{geometry}/iteration/{index}/parameters.csv", f"{resultPath}/{geometry}/iteration/data/{index}")
                    
            displacement, force = read_FD_Curve(f"{simPath}/{geometry}/iteration/{index}/FD_Curve.txt")
            create_FD_Curve_file(f"{resultPath}/{geometry}/iteration/data/{index}", displacement, force)

            new_geom_to_param_FD_Curves[geometry] = {}
            new_geom_to_param_FD_Curves[geometry][paramsTuple] = {}
            new_geom_to_param_FD_Curves[geometry][paramsTuple]['displacement'] = displacement
            new_geom_to_param_FD_Curves[geometry][paramsTuple]['force'] = force
            
        printLog("Saving successfully iteration simulation results", logPath)
        
        return new_geom_to_param_FD_Curves

    def delete_sim_outputs_iteration(self, index):
        simPath = self.info['simPath']
        geometries = self.info['geometries']
        for geometry in geometries:
            if os.path.exists(f"{simPath}/{geometry}/iteration/{index}"):
                shutil.rmtree(f"{simPath}/{geometry}/iteration/{index}")
