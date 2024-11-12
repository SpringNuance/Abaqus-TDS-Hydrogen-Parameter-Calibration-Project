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

class IterationRetrainFramework():

    def __new__(cls, *args, **kwargs):
        print("Creating the Iteration Retrain Framework object")
        instance = super().__new__(cls)
        return instance
    
    def __repr__(self) -> str:
        description = "Iteration Retrain Framework Object"
        return description

    def __init__(self, chosen_project_path, array_job_config, all_paths) -> None:
        
        self.chosen_project_path = chosen_project_path
        self.array_job_config = array_job_config
        self.project_path = all_paths["project_path"]
        self.training_data_path = all_paths["training_data_path"]
        self.models_path = all_paths["models_path"]
        self.scripts_path = all_paths["scripts_path"]

    #################################
    # ITERATION SIMULATION PIPELINE #
    #################################

    def run_iteration_retraining(self, use_referenced_flow_curve, 
                                    current_iteration_index,
                                    previous_iteration_index):
        
        array_job_config = self.array_job_config
        chosen_project_path = self.chosen_project_path
        self.write_shell_script(array_job_config, use_referenced_flow_curve, chosen_project_path,
                                current_iteration_index, previous_iteration_index)
        self.submit_retrain_iteration()

    ################################
    # ITERATION RETRAINING METHODS #
    ################################
    
    def submit_retrain_iteration(self):   

        scripts_path = self.scripts_path

        ########################################
        # CSC command to submit the array jobs #
        ########################################

        # The wait flag is used to wait until all the jobs are finished

        subprocess.run(f"sbatch --wait {scripts_path}/puhti_abaqus_retrain_model.sh", shell=True)
        print("Iteration retraining postprocessing stage finished")
    
    def write_shell_script(self, array_job_config, use_referenced_flow_curve, chosen_project_path,
                            current_iteration_index, previous_iteration_index):
        
        scripts_path = self.scripts_path
        project_path = self.project_path
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
            elif key == "mem":
                script += f"#SBATCH --mem={value}\n"
            elif key == "time":
                script += f"#SBATCH --time={value}\n"
            elif key == "partition":
                script += f"#SBATCH --partition={value}\n"
            elif key == "gres":
                script += f"#SBATCH --gres={value}\n"
            elif key == "account":
                script += f"#SBATCH --account={value}\n"
            elif key == "mail_type":
                script += f"#SBATCH --mail-type={value}\n"
            elif key == "mail_user":
                script += f"#SBATCH --mail-user={value}\n"
        
        # Add environment and module setup commands
        script += """
module load python-data\n"""

        # Change to the work directory
        script += f"""
### Change to the work directory
cd {project_path}\n"""

        # Retraining the LSTM model 
        script += f"""
# Retrain the LSTM model
srun python optimization/LSTM_retrain.py --chosen_project_path {chosen_project_path} --current_iteration_index {current_iteration_index} --previous_iteration_index {previous_iteration_index}\n
"""
        # Retraining the transformer model
        if not use_referenced_flow_curve:
            script += f"""
# Retrain the transformer model
srun python optimization/transformer_retrain.py --chosen_project_path {chosen_project_path} --current_iteration_index {current_iteration_index} --previous_iteration_index {previous_iteration_index}\n
"""
        with open(f"{scripts_path}/puhti_abaqus_retrain_model.sh", "w") as filename:
            filename.write(script)