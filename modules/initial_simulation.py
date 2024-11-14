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
import multiprocessing
from subprocess import Popen

def get_cpu_cores():
    if os.name == 'posix':  # For Unix-based systems (e.g., Linux, macOS)
        try:
            return os.sysconf('SC_NPROCESSORS_ONLN')
        except ValueError:
            pass
    elif os.name == 'nt':  # For Windows
        try:
            return int(os.environ['NUMBER_OF_PROCESSORS'])
        except (ValueError, KeyError):
            pass

    # Fallback option if the above methods fail
    return multiprocessing.cpu_count()

def run_bat_files_parallel(commands, paths):
    processes = []
    for index, command in enumerate(commands):
        process = Popen(command, cwd=paths[index], shell=True)
        processes.append(process)

    # Wait for all processes to finish
    for process in processes:
        process.wait()


class InitialSimFramework():

    def __new__(cls, *args, **kwargs):
        print("Creating the Initial Sim Framework object")
        instance = super().__new__(cls)
        return instance
    
    def __repr__(self) -> str:
        description = ""
        description += "Initial Simulation Framework Object\n"
        description += f"Delete Simulation Outputs: {self.delete_sim}\n" 
        return description
    
    def __init__(self,  param_config, description_properties_dict,
                        delete_sim, all_paths) -> None:
        
        self.param_config = param_config
        self.description_properties_dict = description_properties_dict

        self.delete_sim = delete_sim
        self.project_path = all_paths["project_path"]
        self.log_path = all_paths["log_path"]
        self.results_init_common_path = all_paths["results_init_common_path"]
        self.results_init_data_path = all_paths["results_init_data_path"]
        self.sims_init_path = all_paths["sims_init_path"]
        self.templates_path = all_paths["templates_path"]
        self.scripts_path = all_paths["scripts_path"]
        self.targets_path = all_paths["targets_path"]

    ###############################
    # INITIAL SIMULATION PIPELINE #
    ###############################

    def run_initial_simulations(self, initial_sampled_params_batch, 
                                current_indices, batch_number, input_file_name):
        
        index_params_dict = self.create_index_params_dict(initial_sampled_params_batch, current_indices)
    
        self.preprocess_simulations_initial(index_params_dict, input_file_name)
        self.submit_array_jobs_initial(current_indices)
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
    
    def preprocess_simulations_initial(self, index_params_dict, input_file_name):
        targets_path = self.targets_path
        sims_init_path = self.sims_init_path
        templates_path = self.templates_path
        param_config = self.param_config
        description_properties_dict = self.description_properties_dict

        for index, params_tuple in index_params_dict.items():
            # Create the simulation folder if not exists, else delete the folder and create a new one
            if os.path.exists(f"{sims_init_path}/{index}"):
                shutil.rmtree(f"{sims_init_path}/{index}")
            shutil.copytree(templates_path, f"{sims_init_path}/{index}")

            params_dict = dict(params_tuple)
            #print(params_dict)
            for param_key, param_value in params_dict.items():
                if param_config[param_key]['replace_prop'] == 'thermal':
                    description_properties_dict["hydrogen_diffusion_properties"][param_key]["value"] = str(param_value)
                if param_config[param_key]['replace_prop'] == 'mechanical':
                    description_properties_dict["mechanical_properties"][param_key]["value"] = str(param_value)

            UMAT_PROPERTY, total_UMAT_num_properties = return_UMAT_property(description_properties_dict)
            UMATHT_PROPERTY, total_UMATHT_num_properties = return_UMATHT_property(description_properties_dict)

            replace_TDS_props(f"{sims_init_path}/{index}/{input_file_name}", 
                UMAT_PROPERTY, total_UMAT_num_properties, 
                UMATHT_PROPERTY, total_UMATHT_num_properties)

            check_surface_H = False
            for param_key, param_value in params_dict.items():
                if param_key == "surface_H":
                    check_surface_H = True
                    break

            if check_surface_H:
                surface_H_value = params_dict["surface_H"]
                replace_surface_H(f"{sims_init_path}/{index}/{input_file_name}", surface_H_value)

            # time.sleep(180)

            create_parameter_file(f"{sims_init_path}/{index}", dict(params_tuple))

    def write_paths_initial(self, index_params_dict):
        project_path = self.project_path
        sims_init_path = self.sims_init_path
        scripts_path = self.scripts_path

        with open(f"{scripts_path}/initial_simulation_array_paths.txt", 'w') as filename:
            for index in list(index_params_dict.keys()):
                filename.write(f"{project_path}/{sims_init_path}/{index}\n")
    
    def submit_array_jobs_initial(self, current_indices):
        project_path = self.project_path
        sims_init_path = self.sims_init_path
        log_path = self.log_path
 
        commands = []
        paths = []
        for index in current_indices:
            commands.append("start /wait cmd /c run.bat")
            paths.append(f"{project_path}/{sims_init_path}/{index}")
    
        print_log(f"Number of called subprocesses required: {len(current_indices)}", log_path)
        #time.sleep(180)
        run_bat_files_parallel(commands, paths)
        print_log("Initial simulations for the parameters have finished", log_path)
    
    def postprocess_results_initial(self, batch_number, index_params_dict):

        sims_init_path = self.sims_init_path
        results_init_common_path = self.results_init_common_path
        results_init_data_path = self.results_init_data_path
        targets_path = self.targets_path
        columns_TDS_measurement = pd.read_excel(f"{targets_path}/columns_TDS_measurement.xlsx")["depvar_name"]
        columns_TDS_measurement = columns_TDS_measurement.to_list()
        #print(columns_TDS_measurement)
        #time.sleep(180)
        # The structure of force-displacement curve: dict of (CP params tuple of tuples) -> {force: forceArray , displacement: displacementArray}

        TDS_measurements_batch = {}
        for index, params_tuple in index_params_dict.items():
            if not os.path.exists(f"{results_init_data_path}/{index}"):
                os.mkdir(f"{results_init_data_path}/{index}")
            shutil.copy(f"{sims_init_path}/{index}/TDS_measurement.txt", f"{results_init_data_path}/{index}")
            shutil.copy(f"{sims_init_path}/{index}/parameters.xlsx", f"{results_init_data_path}/{index}")
            shutil.copy(f"{sims_init_path}/{index}/parameters.csv", f"{results_init_data_path}/{index}")

            TDS_measurement_df = read_TDS_measurement(f"{sims_init_path}/{index}/TDS_measurement.txt", columns_TDS_measurement)
            # print(TDS_measurement_df)
            # time.sleep(180)
            # Save it to results_init_data_path
            TDS_measurement_df.to_csv(f"{results_init_data_path}/{index}/TDS_measurement.csv", index=False)
            TDS_measurement_df.to_excel(f"{results_init_data_path}/{index}/TDS_measurement.xlsx", index=False)
            # create_TDS_measurement_file(f"{results_init_data_path}/{index}", displacement, force)

            TDS_measurements_batch[params_tuple] = {}
            for num_measurement, row in enumerate(TDS_measurement_df.iterrows()):
                time = row[1]["time"]
                C_wtppm = row[1]["C_wtppm"]
                C_mol = row[1]["C_mol"]
                if time != 0.0:
                    TDS_measurements_batch[params_tuple][f"measurement_{num_measurement}"] =\
                        {"time": time, "C_wtppm": C_wtppm, "C_mol": C_mol}
            # print(TDS_measurements_batch[params_tuple])
                    
        # Saving force-displacement curve data for current batch
        np.save(f"{results_init_common_path}/TDS_measurements_batch_{batch_number}.npy", TDS_measurements_batch)
        print(f"Saving successfully TDS_measurements_batch_{batch_number}.npy results for batch {batch_number}")
        # time.sleep(180)
        
    def delete_sim_outputs_initial(self, index_params_dict):
        sims_init_path = self.sims_init_path
        for index, params_tuple in index_params_dict.items():
            shutil.rmtree(f"{sims_init_path}/{index}")