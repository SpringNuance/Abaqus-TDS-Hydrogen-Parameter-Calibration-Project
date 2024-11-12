import pandas as pd
import numpy as np
import subprocess
from typing import List, Dict, Any, Tuple, Union
from utils.IO import *
from utils.calculation import *
from utils.hardening_laws import *
import shutil
import random
import time
import os
    
# __new__ is called before __iter__ and then the object is returned

class IterationSimFramework():

    def __new__(cls, *args, **kwargs):
        print("Creating the Iteration Sim Framework object")
        instance = super().__new__(cls)
        return instance
    
    def __repr__(self) -> str:
        description = ""
        description += "Iteration Simulation Framework Object\n"
        description += f"Objective: {self.objective}\n"
        description += f"Delete Simulation Outputs: {self.delete_sim}\n" 
        return description

    def __init__(self, objective, delete_sim, array_job_config,
                 all_paths) -> None:
        
        self.objective = objective
        self.delete_sim = delete_sim
        self.array_job_config = array_job_config
        self.project_path = all_paths["project_path"]
        self.results_iter_common_path = all_paths["results_iter_common_path"]
        self.results_iter_data_path = all_paths["results_iter_data_path"]
        self.sims_iter_path = all_paths["sims_iter_path"]
        self.templates_path = all_paths["templates_path"]
        self.scripts_path = all_paths["scripts_path"]

        self.results_iter_common_objective_path = f"{self.results_iter_common_path}/{objective}"
        self.results_iter_data_objective_path = f"{self.results_iter_data_path}/{objective}"
        self.sims_iter_objective_path = f"{self.sims_iter_path}/{objective}"
        self.templates_objective_path = f"{self.templates_path}/{objective}"

    #################################
    # ITERATION SIMULATION PIPELINE #
    #################################

    def run_iteration_simulations(self, iteration_predicted_params, 
                                true_plastic_strain, iteration_true_stress,
                                prediction_indices, iteration_index, input_file_name):
        
        index_params_dict = self.create_index_params_dict(iteration_predicted_params, prediction_indices)
        index_true_stress_dict = self.create_index_true_stress_dict(iteration_true_stress, prediction_indices)
        array_job_config = self.array_job_config
        
        self.preprocess_simulations_iteration(iteration_index, index_params_dict, index_true_stress_dict, true_plastic_strain, input_file_name)
        self.write_paths_iteration(iteration_index, index_params_dict)
        self.write_array_shell_script(array_job_config, input_file_name)
        self.submit_array_jobs_iteration(index_params_dict)
        self.postprocess_results_iteration(iteration_index, index_params_dict)

        delete_sim = self.delete_sim
        if delete_sim:
            self.delete_sim_outputs_iteration(iteration_index, index_params_dict)

    ################################
    # ITERATION SIMULATION METHODS #
    ################################

    def create_index_params_dict(self, iteration_predicted_params, prediction_indices):
        """
        This function creates a dictionary of prediction index to parameters tuple
        For example at iteration 2, and num_synthetic_predictions is 15, plus one true prediction
        the index_params_dict will be
        {1: ((param1: value), (param2: value, ...), ...), 
         2: ((param1: value), (param2: value, ...), ...),
            ...
         16: ((param1: value), (param2: value, ...), ...)}
        """
        index_params_dict = {}
        for order, params_dict in enumerate(iteration_predicted_params):
            index = str(prediction_indices[order])
            index_params_dict[index] = tuple(params_dict.items())
        return index_params_dict
    
    def create_index_true_stress_dict(self, iteration_true_stress, prediction_indices):
        """
        This function creates a dictionary of index to true stress array
        For example at iteration 2, and num_synthetic_predictions is 15, plus one true prediction
        The index_true_stress_dict will be
        {1: [stress1, stress2, ..., stressN], 
         2: [stress1, stress2, ..., stressN],
            ...
         16: [stress1, stress2, ..., stressN]}
        """
        index_true_stress_dict = {}
        for order, true_stress in enumerate(iteration_true_stress):
            index = str(prediction_indices[order])
            index_true_stress_dict[index] = true_stress
        return index_true_stress_dict

    def preprocess_simulations_iteration(self, iteration_index, index_params_dict, 
                                         index_true_stress_dict, true_plastic_strain, 
                                         input_file_name):
        sims_iter_objective_path = self.sims_iter_objective_path
        templates_objective_path = self.templates_objective_path

        # Create the simulation folder if not exists, else delete the folder and create a new one
        
        if os.path.exists(f"{sims_iter_objective_path}/iteration_{iteration_index}"):
            shutil.rmtree(f"{sims_iter_objective_path}/iteration_{iteration_index}")
        os.makedirs(f"{sims_iter_objective_path}/iteration_{iteration_index}", exist_ok=True)

        for prediction_index, params_tuple in index_params_dict.items():
            
            shutil.copytree(templates_objective_path, f"{sims_iter_objective_path}/iteration_{iteration_index}/prediction_{prediction_index}")
            
            true_stress = index_true_stress_dict[prediction_index]

            replace_flow_curve(f"{sims_iter_objective_path}/iteration_{iteration_index}/prediction_{prediction_index}/{input_file_name}", true_plastic_strain, true_stress)

            create_parameter_file(f"{sims_iter_objective_path}/iteration_{iteration_index}/prediction_{prediction_index}", dict(params_tuple))
            create_flow_curve_file(f"{sims_iter_objective_path}/iteration_{iteration_index}/prediction_{prediction_index}", true_plastic_strain, true_stress)

    def write_paths_iteration(self, iteration_index, index_params_dict):
        project_path = self.project_path
        sims_iter_objective_path = self.sims_iter_objective_path
        scripts_path = self.scripts_path

        with open(f"{scripts_path}/iteration_simulation_array_paths.txt", 'w') as filename:
            for prediction_index in list(index_params_dict.keys()):
                filename.write(f"{project_path}/{sims_iter_objective_path}/iteration_{iteration_index}/prediction_{prediction_index}\n")
    
    def submit_array_jobs_iteration(self, index_params_dict):   
        sims_number = len(index_params_dict)
        scripts_path = self.scripts_path

        print("Initial simulation preprocessing stage starts")
        print(f"Number of jobs required: {sims_number}")

        ########################################
        # CSC command to submit the array jobs #
        ########################################

        # The wait flag is used to wait until all the jobs are finished

        subprocess.run(f"sbatch --wait --array=1-{sims_number} {scripts_path}/puhti_abaqus_array_iteration.sh", shell=True)
        print("Iteration simulation postprocessing stage finished")
    
    def postprocess_results_iteration(self, iteration_index, index_params_dict):

        sims_iter_objective_path = self.sims_iter_objective_path
        results_iter_common_objective_path = self.results_iter_common_objective_path
        results_iter_data_objective_path = self.results_iter_data_objective_path
        
        if os.path.exists(f"{results_iter_data_objective_path}/iteration_{iteration_index}"):
            shutil.rmtree(f"{results_iter_data_objective_path}/iteration_{iteration_index}")
        os.makedirs(f"{results_iter_data_objective_path}/iteration_{iteration_index}", exist_ok=True)

        FD_curves_iteration = {}
        
        for prediction_index, params_tuple in index_params_dict.items():
            
            if not os.path.exists(f"{results_iter_data_objective_path}/iteration_{iteration_index}/prediction_{prediction_index}"):
                os.mkdir(f"{results_iter_data_objective_path}/iteration_{iteration_index}/prediction_{prediction_index}")
            shutil.copy(f"{sims_iter_objective_path}/iteration_{iteration_index}/prediction_{prediction_index}/FD_curve.txt", 
                        f"{results_iter_data_objective_path}/iteration_{iteration_index}/prediction_{prediction_index}")
            shutil.copy(f"{sims_iter_objective_path}/iteration_{iteration_index}/prediction_{prediction_index}/parameters.xlsx", 
                        f"{results_iter_data_objective_path}/iteration_{iteration_index}/prediction_{prediction_index}")
            shutil.copy(f"{sims_iter_objective_path}/iteration_{iteration_index}/prediction_{prediction_index}/parameters.csv", 
                        f"{results_iter_data_objective_path}/iteration_{iteration_index}/prediction_{prediction_index}")
            shutil.copy(f"{sims_iter_objective_path}/iteration_{iteration_index}/prediction_{prediction_index}/flow_curve.xlsx", 
                        f"{results_iter_data_objective_path}/iteration_{iteration_index}/prediction_{prediction_index}")
            shutil.copy(f"{sims_iter_objective_path}/iteration_{iteration_index}/prediction_{prediction_index}/flow_curve.csv", 
                        f"{results_iter_data_objective_path}/iteration_{iteration_index}/prediction_{prediction_index}")

            displacement, force = read_FD_curve(f"{sims_iter_objective_path}/iteration_{iteration_index}/prediction_{prediction_index}/FD_curve.txt")
            create_FD_curve_file(f"{results_iter_data_objective_path}/iteration_{iteration_index}/prediction_{prediction_index}", displacement, force)

            FD_curves_iteration[params_tuple] = {"displacement": displacement, "force": force}
                    
        # Saving force-displacement curve data for current iteration
        np.save(f"{results_iter_common_objective_path}/FD_curves_iteration_{iteration_index}.npy", FD_curves_iteration)
        print(f"Saving successfully FD_curves_iteration_{iteration_index}.npy results for iteration {iteration_index} of {self.objective}")
    
    def delete_sim_outputs_iteration(self, iteration_index, index_params_dict):
        sims_iter_objective_path = self.sims_iter_objective_path
        for prediction_index, params_tuple in index_params_dict.items():
            shutil.rmtree(f"{sims_iter_objective_path}/iteration_{iteration_index}/prediction_{prediction_index}")

    def write_array_shell_script(self, array_job_config, input_file_name):
        input_file_name_without_extension = input_file_name.split(".")[0]
        scripts_path = self.scripts_path
        # Start building the shell script with the header and shebang
        script = "#!/bin/bash -l\n"
        script += "# Author: Xuan Binh\n"
        
        # Add SBATCH directives based on the configuration dictionary
        for key, value in array_job_config.items():
            if key == "job_name":
                script += f"#SBATCH --job-name={value}\n"
            elif key == "nodes":
                script += f"#SBATCH --nodes={value}\n"
            elif key == "ntasks":
                script += f"#SBATCH --ntasks={value}\n"
            elif key == "cpus_per_task":
                script += f"#SBATCH --cpus-per-task={value}\n"
            elif key == "time":
                script += f"#SBATCH --time={value}\n"
            elif key == "partition":
                script += f"#SBATCH --partition={value}\n"
            elif key == "account":
                script += f"#SBATCH --account={value}\n"
            elif key == "mail_type":
                script += f"#SBATCH --mail-type={value}\n"
            elif key == "mail_user":
                script += f"#SBATCH --mail-user={value}\n"
        
        # Add environment and module setup commands
        script += """
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
unset SLURM_GTIDS
module purge
module load abaqus\n"""

        # Change to the work directory
        script += f"""
### Change to the work directory
fullpath=$(sed -n ${{SLURM_ARRAY_TASK_ID}}p {scripts_path}/iteration_simulation_array_paths.txt) 
cd ${{fullpath}}\n"""

        # Construct the Abaqus command with dynamic CPU allocation
        script += f"""
CPUS_TOTAL=$(( $SLURM_NTASKS*$SLURM_CPUS_PER_TASK ))

abaqus job={input_file_name_without_extension} input={input_file_name} cpus=$CPUS_TOTAL -verbose 2 interactive\n"""

        # Post-processing command
        script += """
# run postprocess.py after the simulation completes
abaqus cae noGUI=postprocess.py\n"""
                
        with open(f"{scripts_path}/puhti_abaqus_array_iteration.sh", 'w') as filename:
            filename.write(script)