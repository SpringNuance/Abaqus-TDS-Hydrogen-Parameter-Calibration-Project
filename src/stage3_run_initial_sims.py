import numpy as np

from modules.initial_simulation import *
from utils.IO import *
from utils.calculation import *
from src.stage1_global_configs import main_global_configs
from src.stage2_prepare_common_data import main_prepare_common_data
from math import *
import os


def main_run_initial_sims(global_configs, stage2_outputs):

    # ================================ #
    # Stage 3: Run initial simulations #
    # ================================ #
    
    initial_sim_config = global_configs['initial_sim_config']
    
    max_concurrent_samples = initial_sim_config['max_concurrent_samples']
    input_file_name = initial_sim_config['input_file_name']
    delete_sim = initial_sim_config['delete_sim']
    
    all_paths = global_configs['all_paths']
    log_path = all_paths['log_path']
    targets_path = all_paths['targets_path']
    results_init_common_path = all_paths['results_init_common_path']
    param_config = global_configs['param_config']


    print_log("\n========================================", log_path)
    print_log("= Stage 3: Running initial simulations =", log_path)
    print_log("========================================\n", log_path)

    initial_sampled_parameters = stage2_outputs.get('initial_sampled_parameters', None)
    TDS_measurements = stage2_outputs.get('TDS_measurements', None)
    description_properties_dict = stage2_outputs.get('description_properties_dict', None)
    
    if initial_sampled_parameters is None:
        raise ValueError("Initial sampled parameters are not provided")

    num_samples = len(initial_sampled_parameters)

    num_batches = int(ceil(num_samples / max_concurrent_samples))
    
    print_log(f"Number of samples: {num_samples}", log_path)
    print_log(f"Max concurrent samples: {max_concurrent_samples}", log_path)
    print_log(f"Number of batches: {num_batches}", log_path)

    #time.sleep(180)

    if os.path.exists(f"{results_init_common_path}/TDS_measurements.npy"):
        print_log("=======================================================================", log_path)
        print_log(f"TDS_measurements.npy already exist.", log_path)
    else:
        # This initial sim framework handles everything from preprocessing to postprocessing 
        initial_sim_framework = InitialSimFramework(param_config, description_properties_dict, 
                                                    delete_sim=delete_sim,
                                                    all_paths=all_paths) 
    
        print_log("=======================================================================", log_path)
        print_log(f"TDS_measurements.npy does not exist", log_path)
        
        # This loop also works for case when there is only 1 batch
        for sim_index in range(0, num_samples, max_concurrent_samples):
            batch_number = int(sim_index/max_concurrent_samples + 1)
            current_indices = range(sim_index + 1, sim_index + max_concurrent_samples + 1)
        
            if not os.path.exists(f"{results_init_common_path}/TDS_measurements_batch_{batch_number}.npy"):
                print_log("=======================================================================", log_path)
                print_log(f"There is no TDS_measurements_batch_{batch_number}.npy", log_path)
                print_log(f"Program starts running the initial simulations of batch number {batch_number}", log_path)   
                print_log(f"The current indices first and last values are {current_indices[0]} and {current_indices[-1]}", log_path)
                
                
                if sim_index + max_concurrent_samples > len(initial_sampled_parameters):
                    initial_sampled_parameters_batch = initial_sampled_parameters[sim_index:]
                else:
                    initial_sampled_parameters_batch = initial_sampled_parameters[sim_index : sim_index + max_concurrent_samples]
                initial_sim_framework.run_initial_simulations(initial_sampled_parameters_batch, 
                                                            current_indices, batch_number, input_file_name)
                
                print_log(f"Initial simulations of batch {batch_number} have completed", log_path)
                # time.sleep(180)
            else: 
                print_log("=======================================================================", log_path)
                print_log(f"TDS_measurements_batch_{batch_number}.npy already exists", log_path)
            
        # Now we combine the batches of simulations together into TDS_measurements.npy
        print_log("=======================================================================", log_path)
        print_log(f"Start concatenating the batches of initial guesses together", log_path)

        TDS_measurements = {}

        for i in range(0, num_samples, max_concurrent_samples):
            batch_number = int(i/max_concurrent_samples + 1)
            TDS_measurements_batch = np.load(f"{results_init_common_path}/TDS_measurements_batch_{batch_number}.npy", allow_pickle=True).tolist()
            TDS_measurements.update(TDS_measurements_batch)
        np.save(f"{results_init_common_path}/TDS_measurements.npy", TDS_measurements)
        num_simulations = len(TDS_measurements)
        print_log(f"Finish concatenating the batches of initial guesses together", log_path)
        print_log(f"Number of initial simulations: {num_simulations} FD curves", log_path)

    initial_sim_TDS_measurements = np.load(f"{results_init_common_path}/TDS_measurements.npy", allow_pickle=True).tolist()
    stage3_outputs = {
        'initial_sim_TDS_measurements': initial_sim_TDS_measurements
    }

    return stage3_outputs

if __name__ == "__main__":
    global_configs = main_global_configs()
    stage2_outputs = main_prepare_common_data(global_configs)
    main_run_initial_sims(global_configs, stage2_outputs)