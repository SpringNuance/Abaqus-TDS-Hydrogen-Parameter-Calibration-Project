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
    input_file_names = initial_sim_config['input_file_names']
    delete_sim = initial_sim_config['delete_sim']
    initial_array_job_config = initial_sim_config['array_job_config']
    
    objectives = global_configs['objectives']
    all_paths = global_configs['all_paths']
    log_path = all_paths['log_path']
    targets_path = all_paths['targets_path']
    results_init_common_path = all_paths['results_init_common_path']

    print_log("\n========================================", log_path)
    print_log("= Stage 3: Running initial simulations =", log_path)
    print_log("========================================\n", log_path)

    initial_sampled_parameters = stage2_outputs.get('initial_sampled_parameters', None)
    true_plastic_strain = stage2_outputs.get('true_plastic_strain', None)
    initial_sampled_true_stress = stage2_outputs.get('initial_sampled_true_stress', None)

    if initial_sampled_parameters is None:
        raise ValueError("Initial sampled parameters are not provided")
    if true_plastic_strain is None:
        raise ValueError("True plastic strain is not provided")
    if initial_sampled_true_stress is None:
        raise ValueError("Initial sampled true stress is not provided")
    num_samples = len(initial_sampled_parameters)

    num_batches = int(ceil(num_samples / max_concurrent_samples))
    
    print_log(f"Number of samples: {num_samples}", log_path)
    print_log(f"Max concurrent samples: {max_concurrent_samples}", log_path)
    print_log(f"Number of batches: {num_batches}", log_path)
    print_log(f"Total initial sims: {num_samples} x {len(objectives)} = {num_samples * len(objectives)} FD curves", log_path)
    
    # How does this workflow work?
    # Because the CSC system has a limit on a number of jobs for each user 
    # and the number of Abaqus licenses are also limited (CSC server has at most 600 licenses)
    # We need to run the initial simulations in batches to avoid requesting too much resources in an array job
    # The number of batches is calculated by the number of initial samples divided by the maximum number of concurrent samples
    # Let's say there are 7 objectives, and we need to obtain 256 FD curves for each objective to train the Seq2Seq model
    # The total number of FD curves is 7 x 256 = 1792 FD curves
    # If the maximum number of concurrent samples is 64, then we need to run 1792 / 64 = 28 batches of simulations

    # time.sleep(180)

    for objective in objectives:    

        if os.path.exists(f"{results_init_common_path}/{objective}/FD_curves.npy"):
            print_log("=======================================================================", log_path)
            print_log(f"FD_curves.npy for {objective} already exist.", log_path)
        else:

            # This initial sim framework handles everything from preprocessing to postprocessing 
            initial_sim_framework = InitialSimFramework(objective=objective,
                                                        delete_sim=delete_sim,
                                                        array_job_config=initial_array_job_config,
                                                        all_paths=all_paths) 
        
            print_log("=======================================================================", log_path)
            print_log(f"FD_curves.npy for {objective} does not exist", log_path)
            
            # This loop also works for case when there is only 1 batch
            for sim_index in range(0, num_samples, max_concurrent_samples):
                batch_number = int(sim_index/max_concurrent_samples + 1)
                current_indices = range(sim_index + 1, sim_index + max_concurrent_samples + 1)
            
                if not os.path.exists(f"{results_init_common_path}/{objective}/FD_curves_batch_{batch_number}.npy"):
                    print_log("=======================================================================", log_path)
                    print_log(f"There is no FD_curves_batch_{batch_number}.npy for {objective}", log_path)
                    print_log(f"Program starts running the initial simulations of batch number {batch_number} for {objective}", log_path)   
                    print_log(f"The current indices first and last values are {current_indices[0]} and {current_indices[-1]}", log_path)
                    
                    if sim_index + max_concurrent_samples > len(initial_sampled_parameters):
                        initial_sampled_parameters_batch = initial_sampled_parameters[sim_index:]
                        true_stress_batch = initial_sampled_true_stress[sim_index:]
                    else:
                        initial_sampled_parameters_batch = initial_sampled_parameters[sim_index : sim_index + max_concurrent_samples]
                        true_stress_batch = initial_sampled_true_stress[sim_index : sim_index + max_concurrent_samples]
                    input_file_name = input_file_names[objective]
                    initial_sim_framework.run_initial_simulations(initial_sampled_parameters_batch, 
                                                                  true_plastic_strain, true_stress_batch,
                                                                  current_indices, batch_number, input_file_name)
                    
                    print_log(f"Initial simulations of batch {batch_number} for {objective} have completed", log_path)
                else: 
                    print_log("=======================================================================", log_path)
                    print_log(f"FD_curves_batch_{batch_number}.npy for {objective} already exists", log_path)
                
            # Now we combine the batches of simulations together into FD_curves.npy
            print_log("=======================================================================", log_path)
            print_log(f"Start concatenating the batches of initial guesses together", log_path)

            FD_curves = {}

            for i in range(0, num_samples, max_concurrent_samples):
                batch_number = int(i/max_concurrent_samples + 1)
                FD_curves_batch = np.load(f"{results_init_common_path}/{objective}/FD_curves_batch_{batch_number}.npy", allow_pickle=True).tolist()
                FD_curves.update(FD_curves_batch)
            np.save(f"{results_init_common_path}/{objective}/FD_curves.npy", FD_curves)
            num_simulations = len(FD_curves)
            print_log(f"Finish concatenating the batches of initial guesses together for {objective}", log_path)
            print_log(f"Number of initial simulations for {objective}: {num_simulations} FD curves", log_path)
 
    # FD_curves_interpolated.npy for each objective
    # initial_FD_curves_interpolated_combined for all

    from utils.calculation import interpolating_force

    initial_FD_curves_interpolated_combined = {}

    for i, objective in enumerate(objectives):
        if not os.path.exists(f"{results_init_common_path}/{objective}/FD_curves_interpolated.npy"): 
            FD_curves_interpolated = {}
            FD_curves_batch = np.load(f"{results_init_common_path}/{objective}/FD_curves.npy", allow_pickle=True).tolist()
            interpolated_displacement_pd = pd.read_excel(f"{targets_path}/{objective}/interpolated_displacement.xlsx")           
            interpolated_displacement = interpolated_displacement_pd['displacement/m'].to_numpy()

            for params_tuple, FD_curve in FD_curves_batch.items():
                sim_displacement = FD_curve['displacement']
                sim_force = FD_curve['force']
                interpolated_force = interpolating_force(sim_displacement, sim_force, interpolated_displacement)
                FD_curves_interpolated[params_tuple] = {'displacement': interpolated_displacement, 'force': interpolated_force}
            
            initial_FD_curves_interpolated_combined[objective] = FD_curves_interpolated
            np.save(f"{results_init_common_path}/{objective}/FD_curves_interpolated.npy", FD_curves_interpolated)
            print_log(f"Interpolated FD curves for {objective} are saved", log_path)
        else:
            print_log(f"Interpolated FD curves for {objective} already exist", log_path)
            initial_FD_curves_interpolated_combined[objective] = np.load(f"{results_init_common_path}/{objective}/FD_curves_interpolated.npy", allow_pickle=True).tolist()

    stage3_outputs = {
        'initial_FD_curves_interpolated_combined': initial_FD_curves_interpolated_combined
    }

    return stage3_outputs

if __name__ == "__main__":
    global_configs = main_global_configs()
    stage2_outputs = main_prepare_common_data(global_configs)
    main_run_initial_sims(global_configs, stage2_outputs)