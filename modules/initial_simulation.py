import pandas as pd
import numpy as np
import subprocess
from typing import List, Dict, Any, Tuple, Union
from utils.IO import *
from utils.calculation import *
from utils.hardening_laws import *
import sys
import shutil
import random
import time
import os

class InitialSimFramework():

    def __new__(cls, *args, **kwargs):
        print("Creating the Initial Sim Framework object")
        instance = super().__new__(cls)
        return instance
    
    def __repr__(self) -> str:
        description = ""
        description += "Initial Simulation Framework Object\n"
        description += f"Objective: {self.objective}\n"
        description += f"Delete Simulation Outputs: {self.delete_sim}\n" 
        return description
    
    def __init__(self, objective, delete_sim, array_job_config,
                 all_paths) -> None:
        
        self.objective = objective
        self.delete_sim = delete_sim
        self.array_job_config = array_job_config
        self.project_path = all_paths["project_path"]
        self.results_init_common_path = all_paths["results_init_common_path"]
        self.results_init_data_path = all_paths["results_init_data_path"]
        self.sims_init_path = all_paths["sims_init_path"]
        self.templates_path = all_paths["templates_path"]
        self.scripts_path = all_paths["scripts_path"]

        self.results_init_common_objective_path = f"{self.results_init_common_path}/{objective}"
        self.results_init_data_objective_path = f"{self.results_init_data_path}/{objective}"
        self.sims_init_objective_path = f"{self.sims_init_path}/{objective}"
        self.templates_objective_path = f"{self.templates_path}/{objective}"

    ###############################
    # INITIAL SIMULATION PIPELINE #
    ###############################

    def run_initial_simulations(self, initial_sampled_params_batch, 
                                true_plastic_strain, true_stress_batch,
                                current_indices, batch_number, input_file_name):
        
        index_params_dict = self.create_index_params_dict(initial_sampled_params_batch, current_indices)
        index_true_stress_dict = self.create_index_true_stress_dict(true_stress_batch, current_indices)
        array_job_config = self.array_job_config
        
        self.preprocess_simulations_initial(index_params_dict, index_true_stress_dict, true_plastic_strain, input_file_name)
        self.write_paths_initial(index_params_dict)
        self.write_array_shell_script(array_job_config, input_file_name)
        self.submit_array_jobs_initial(index_params_dict)
        self.postprocess_results_initial(batch_number, index_params_dict)

        delete_sim = self.delete_sim
        if delete_sim:
            self.delete_sim_outputs_initial(index_params_dict)

    ##############################
    # INITIAL SIMULATION METHODS #
    ##############################

    # Clarification
    # index is the index passed from the current indices of the current batch
    # For example, if max_concurrent_samples is 64, and current batch is 2,
    # then the current indices will be [65, 66, 67, ..., 128]
    # On the other hand, order is always the count from 0 to max_concurrent_samples - 1 for one batch
          
    def create_index_params_dict(self, initial_sampled_params_batch, current_indices):
        """
        This function creates a dictionary of index to parameters tuple
        For example at batch 2, max_concurrent_samples is 64
        The index_params_dict will be
        {65: ((param1: value), (param2: value, ...), ...), 
         66: ((param1: value), (param2: value, ...), ...),
            ...
         128: ((param1: value), (param2: value, ...), ...)}
        """
        index_params_dict = {}
        for order, params_dict in enumerate(initial_sampled_params_batch):
            index = str(current_indices[order])
            index_params_dict[index] = tuple(params_dict.items())
        return index_params_dict
    
    def create_index_true_stress_dict(self, true_stress_batch, current_indices):
        """
        This function creates a dictionary of index to true stress array
        For example at batch 2, max_concurrent_samples is 64
        The index_true_stress_dict will be
        {65: [stress1, stress2, ..., stressN], 
         66: [stress1, stress2, ..., stressN],
            ...
         128: [stress1, stress2, ..., stressN]}
        """
        index_true_stress_dict = {}
        for order, true_stress in enumerate(true_stress_batch):
            index = str(current_indices[order])
            index_true_stress_dict[index] = true_stress
        return index_true_stress_dict
    
    def preprocess_simulations_initial(self, index_params_dict, index_true_stress_dict, true_plastic_strain, input_file_name):
        sims_init_objective_path = self.sims_init_objective_path
        templates_objective_path = self.templates_objective_path

        for index, params_tuple in index_params_dict.items():
            # Create the simulation folder if not exists, else delete the folder and create a new one
            if os.path.exists(f"{sims_init_objective_path}/{index}"):
                shutil.rmtree(f"{sims_init_objective_path}/{index}")
            shutil.copytree(templates_objective_path, f"{sims_init_objective_path}/{index}")
            
            true_stress = index_true_stress_dict[index]

            replace_flow_curve(f"{sims_init_objective_path}/{index}/{input_file_name}", true_plastic_strain, true_stress)

            create_parameter_file(f"{sims_init_objective_path}/{index}", dict(params_tuple))
            create_flow_curve_file(f"{sims_init_objective_path}/{index}", true_plastic_strain, true_stress)

    def write_paths_initial(self, index_params_dict):
        project_path = self.project_path
        sims_init_objective_path = self.sims_init_objective_path
        scripts_path = self.scripts_path

        with open(f"{scripts_path}/initial_simulation_array_paths.txt", 'w') as filename:
            for index in list(index_params_dict.keys()):
                filename.write(f"{project_path}/{sims_init_objective_path}/{index}\n")
    
    def submit_array_jobs_initial(self, index_params_dict):   
        sims_number = len(index_params_dict)
        scripts_path = self.scripts_path

        print("Initial simulation preprocessing stage starts")
        print(f"Number of jobs required: {sims_number}")

        ########################################
        # CSC command to submit the array jobs #
        ########################################

        # The wait flag is used to wait until all the jobs are finished

        subprocess.run(f"sbatch --wait --array=1-{sims_number} {scripts_path}/puhti_abaqus_array_initial.sh", shell=True)
        print("Initial simulation postprocessing stage finished")
    
    def postprocess_results_initial(self, batch_number, index_params_dict):

        sims_init_objective_path = self.sims_init_objective_path
        results_init_common_objective_path = self.results_init_common_objective_path
        results_init_data_objective_path = self.results_init_data_objective_path
        
        # The structure of force-displacement curve: dict of (CP params tuple of tuples) -> {force: forceArray , displacement: displacementArray}

        FD_curves_batch = {}
        for index, params_tuple in index_params_dict.items():
            if not os.path.exists(f"{results_init_data_objective_path}/{index}"):
                os.mkdir(f"{results_init_data_objective_path}/{index}")
            shutil.copy(f"{sims_init_objective_path}/{index}/FD_curve.txt", f"{results_init_data_objective_path}/{index}")
            shutil.copy(f"{sims_init_objective_path}/{index}/parameters.xlsx", f"{results_init_data_objective_path}/{index}")
            shutil.copy(f"{sims_init_objective_path}/{index}/parameters.csv", f"{results_init_data_objective_path}/{index}")
            shutil.copy(f"{sims_init_objective_path}/{index}/flow_curve.xlsx", f"{results_init_data_objective_path}/{index}")
            shutil.copy(f"{sims_init_objective_path}/{index}/flow_curve.csv", f"{results_init_data_objective_path}/{index}")

            displacement, force = read_FD_curve(f"{sims_init_objective_path}/{index}/FD_curve.txt")
            create_FD_curve_file(f"{results_init_data_objective_path}/{index}", displacement, force)

            FD_curves_batch[params_tuple] = {"displacement": displacement, "force": force}
                    
        # Saving force-displacement curve data for current batch
        np.save(f"{results_init_common_objective_path}/FD_curves_batch_{batch_number}.npy", FD_curves_batch)
        print(f"Saving successfully FD_curves_batch_{batch_number}.npy results for batch {batch_number} of {self.objective}")
    
    def delete_sim_outputs_initial(self, index_params_dict):
        sims_init_objective_path = self.sims_init_objective_path
        for index, params_tuple in index_params_dict.items():
            shutil.rmtree(f"{sims_init_objective_path}/{index}")

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
fullpath=$(sed -n ${{SLURM_ARRAY_TASK_ID}}p {scripts_path}/initial_simulation_array_paths.txt) 
cd ${{fullpath}}\n"""

        # Construct the Abaqus command with dynamic CPU allocation
        script += f"""
CPUS_TOTAL=$(( $SLURM_NTASKS*$SLURM_CPUS_PER_TASK ))

abaqus job={input_file_name_without_extension} input={input_file_name} cpus=$CPUS_TOTAL -verbose 2 interactive\n"""

        # Post-processing command
        script += """
# run postprocess.py after the simulation completes
abaqus cae noGUI=postprocess.py\n"""
                
        with open(f"{scripts_path}/puhti_abaqus_array_initial.sh", 'w') as filename:
            filename.write(script)