import numpy as np
import time
from utils.IO import *
from utils.calculation import *
from modules.stoploss import *
from utils.sampling import *
from math import *
import os
import copy
import torch

from src.stage1_global_configs import * 
from src.stage2_prepare_common_data import *
from src.stage3_run_initial_sims import *
from itertools import product

def main_prepare_sim_data(global_configs, stage2_outputs):

    # ------------------------------------- #
    #  Stage 4: Preparing simulations data  #
    # ------------------------------------- #
    
    all_paths = global_configs["all_paths"]
    num_measurements = global_configs["num_measurements"]
    
    results_init_common_path = all_paths["results_init_common_path"]
    targets_path = all_paths["targets_path"]
    training_data_path = all_paths["training_data_path"]
    log_path = all_paths["log_path"]

    print_log("\n===================================", log_path)
    print_log("= Stage 4: Prepare simulation data =", log_path)
    print_log("====================================\n", log_path)

    ################################
    # PROCESSING INITIAL SIMS DATA #
    ################################

    results_init_common_path = all_paths["results_init_common_path"]
    results_iter_common_path = all_paths["results_iter_common_path"]
    training_data_path = all_paths["training_data_path"]
    targets_path = all_paths["targets_path"]
    param_config = global_configs["param_config"]
    model_config = global_configs["model_config"]
    loss_function = global_configs["stop_loss_config"]["loss_function"]
    augmented_spacing_points = global_configs["initial_sim_config"]["augmented_spacing_points"]

    target_TDS_measurements = stage2_outputs["target_TDS_measurements"]

    ###########################
    # INITIAL SIMULATION DATA #
    ###########################

    # Loading the TDS_measurements.npy
    initial_sim_TDS_measurements = np.load(f"{results_init_common_path}/TDS_measurements.npy", allow_pickle=True).tolist()

    initial_features_unnormalized = []

    for index, param_key in enumerate(param_config.keys()):
        param_values_list = np.array([params_tuple[index][1] for params_tuple in initial_sim_TDS_measurements.keys()])
        initial_features_unnormalized.append(param_values_list)

    initial_features_unnormalized = np.array(initial_features_unnormalized).T
 
    # Now we need to normalize the initial_features_unnormalized and initial_labels

    initial_features_normalized = normalize_points(initial_features_unnormalized, param_config)

    initial_labels = []
    for params_tuple, sim_measurements in initial_sim_TDS_measurements.items():
        current_loss = 0
        target_values = []
        sim_values = []  
        for sim_measurement in sim_measurements.values():
            sim_measurement_time = sim_measurement["time"]
            for target_measurement in target_TDS_measurements.values():
                target_measurement_time = target_measurement["time"]
                if sim_measurement_time == target_measurement_time:
                    target_values.append(target_measurement["C_mol"])
                    sim_values.append(sim_measurement["C_mol"])
                    break

        target_values = np.array(target_values)
        sim_values = np.array(sim_values)

        initial_labels.append(- calculate_loss(target_values, sim_values, loss_function))
    
    initial_labels = np.array(initial_labels)

    print_log(f"Shape of initial_features_normalized: {initial_features_normalized.shape}", log_path)
    print_log(f"Shape of initial_features_unnormalized: {initial_features_unnormalized.shape}", log_path)
    print_log(f"Shape of initial_labels: {initial_labels.shape}", log_path)

    # print(f"Initial features normalized: {initial_features_normalized}")
    # print(f"Initial features unnormalized: {initial_features_unnormalized}")
    # print(f"Initial labels: {initial_labels}")


    if "surface_H" in param_config.keys():
        print_log(f"There is surface_H in the param_config", log_path)
        print_log(f"Calculating the known loss when H_surface = 0", log_path)
        # sim_values would all be zeros
        
        target_values = np.array([value["C_mol"] for value in target_TDS_measurements.values()])
        known_loss = calculate_loss(target_values, np.zeros_like(sim_values), loss_function)
        print_log(f"Known loss: {known_loss}", log_path)
        num_params = len(param_config.keys())
        # Now we augment the initial_features_normalized, initial_features_unnormalized like this
        # For surface_H, we add a new column of zeros
        # For other params, we add a linearly spacing of 0 to 1 with N points as a cartesian product
        # Example of 2 params: H_surface and DL, N = 3. This gives us 3 points
        # H_surface = [0, 0, 0], DL = [0, 0.5, 1]
        # Example of 3 params: H_surface, DL, and eqplas, N = 3. This gives us 9 points
        # H_surface = [0, 0, 0, 0, 0, 0, 0, 0, 0], 
        # DL = [0, 0.5, 1, 0, 0.5, 1, 0, 0.5, 1], 
        # eqplas = [0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1]

        # Get the number of parameters and initialize lists for augmented features and labels
        num_params = len(param_config.keys())
        param_names = list(param_config.keys())
        
        # Generate a linearly spaced array for each parameter except 'surface_H'
        unnormalized_linear_spaces = {}
        normalized_linear_spaces = {}
        for param_name in param_names:
            if param_name == 'surface_H':
                # Fix surface_H at zero
                unnormalized_linear_spaces[param_name] = np.zeros(augmented_spacing_points)
                normalized_linear_spaces[param_name] = np.zeros(augmented_spacing_points)
            else:
                # Linearly space other parameters between 0 and 1
                unnormalized_linear_spaces[param_name] = np.linspace(param_config[param_name]["lower"], 
                                                                     param_config[param_name]["upper"], 
                                                                     augmented_spacing_points)
                normalized_linear_spaces[param_name] = np.linspace(0, 1, augmented_spacing_points)
        
        # Separate parameters that should be used in the Cartesian product
        other_param_names = [param for param in param_names if param != 'surface_H']
        
        # Create a Cartesian product only for the other parameters
        unnormalized_cartesian_product = product(*[unnormalized_linear_spaces[param] for param in other_param_names])
        normalized_cartesian_product = product(*[normalized_linear_spaces[param] for param in other_param_names])
        
        # Convert Cartesian products to arrays
        augmented_features_unnormalized = np.array(list(unnormalized_cartesian_product))
        augmented_features_normalized = np.array(list(normalized_cartesian_product))
        
        # Add a zero column for surface_H at the beginning of both augmented arrays
        surface_H_zeros = np.zeros((augmented_features_normalized.shape[0], 1))
        augmented_features_unnormalized = np.hstack((surface_H_zeros, augmented_features_unnormalized))
        augmented_features_normalized = np.hstack((surface_H_zeros, augmented_features_normalized))
        
        # Set the augmented labels to the known_loss for each new point
        augmented_labels = np.full((augmented_features_normalized.shape[0],), - known_loss)

        # Concatenate the augmented data to the initial data
        initial_features_normalized_augmented = np.vstack((initial_features_normalized, augmented_features_normalized))
        initial_features_unnormalized_augmented  = np.vstack((initial_features_unnormalized, augmented_features_unnormalized))
        initial_labels_augmented = np.concatenate((initial_labels, augmented_labels))

        print_log(f"Shape of initial_features_normalized_augmented: {initial_features_normalized_augmented.shape}", log_path)
        print_log(f"Shape of initial_features_unnormalized_augmented: {initial_features_unnormalized_augmented.shape}", log_path)
        print_log(f"Shape of initial_labels_augmented: {initial_labels_augmented.shape}", log_path)

        # print(f"Initial features normalized augmented: {initial_features_normalized_augmented}")
        # print(f"Initial features unnormalized augmented: {initial_features_unnormalized_augmented}")
        # print(f"Initial labels augmented: {initial_labels_augmented}")

    # time.sleep(180)

    # Save the initial data
    np.save(f"{training_data_path}/initial_features_normalized.npy", initial_features_normalized)
    np.save(f"{training_data_path}/initial_features_unnormalized.npy", initial_features_unnormalized)
    np.save(f"{training_data_path}/initial_labels.npy", initial_labels)

    if "surface_H" in param_config.keys():
        np.save(f"{training_data_path}/initial_features_normalized_augmented.npy", initial_features_normalized_augmented)
        np.save(f"{training_data_path}/initial_features_unnormalized_augmented.npy", initial_features_unnormalized_augmented)
        np.save(f"{training_data_path}/initial_labels_augmented.npy", initial_labels_augmented)
    
    ####################################
    # FIND THE CURRENT ITERATION INDEX #
    ####################################

    current_iteration_index = 1
    while os.path.exists(f"{results_iter_common_path}/iteration_{current_iteration_index}"):
        current_iteration_index += 1
    current_iteration_index -= 1
    
    ########################################
    # COMBINING INITIAL AND ITERATION DATA #
    ########################################

    if current_iteration_index == 0:
        print_log(f"No iteration data found - Loading only the initial data", log_path)
        combined_features_normalized = initial_features_normalized
        combined_features_unnormalized = initial_features_unnormalized
        combined_labels = initial_labels
        if "surface_H" in param_config.keys():
            combined_features_normalized_augmented = initial_features_normalized_augmented
            combined_features_unnormalized_augmented = initial_features_unnormalized_augmented
            combined_labels_augmented = initial_labels_augmented
    else:
        print_log(f"The current iteration index is {current_iteration_index}", log_path)
        print_log(f"Combining the initial and iteration data into combined data", log_path)

        if os.path.exists(f"{results_iter_common_path}/iteration_common/TDS_measurements.npy"):
            print_log(f"Loading the iteration TDS measurements", log_path)
            
            # Loading the TDS_measurements.npy
            iteration_sim_TDS_measurements = np.load(f"{results_init_common_path}/iteration_common/TDS_measurements.npy", allow_pickle=True).tolist()

            iteration_features_unnormalized = []

            for index, param_key in enumerate(param_config.keys()):
                param_values_list = np.array([params_tuple[index][1] for params_tuple in iteration_sim_TDS_measurements.keys()])
                iteration_features_unnormalized.append(param_values_list)

            iteration_features_unnormalized = np.array(iteration_features_unnormalized).T
        
            # Now we need to normalize the iteration_features_unnormalized and iteration_labels

            iteration_features_normalized = normalize_points(iteration_features_unnormalized, param_config)

            iteration_labels = []
            for params_tuple, sim_measurements in iteration_sim_TDS_measurements.items():
                current_loss = 0
                target_values = []
                sim_values = []  
                for sim_measurement in sim_measurements.values():
                    sim_measurement_time = sim_measurement["time"]
                    for target_measurement in target_TDS_measurements.values():
                        target_measurement_time = target_measurement["time"]
                        if sim_measurement_time == target_measurement_time:
                            target_values.append(target_measurement["C_mol"])
                            sim_values.append(sim_measurement["C_mol"])
                            break

                target_values = np.array(target_values)
                sim_values = np.array(sim_values)

                iteration_labels.append(- calculate_loss(target_values, sim_values, loss_function))
            
            iteration_labels = np.array(iteration_labels)

            print_log(f"Shape of iteration_features_normalized: {iteration_features_normalized.shape}", log_path)
            print_log(f"Shape of iteration_features_unnormalized: {iteration_features_unnormalized.shape}", log_path)
            print_log(f"Shape of iteration_labels: {iteration_labels.shape}", log_path)
        
            # Save the iteration data
            np.save(f"{training_data_path}/iteration_features_normalized.npy", iteration_features_normalized)
            np.save(f"{training_data_path}/iteration_features_unnormalized.npy", iteration_features_unnormalized)
            np.save(f"{training_data_path}/iteration_labels.npy", iteration_labels)

    
            combined_features_normalized = np.vstack((initial_features_normalized, iteration_features_normalized))
            combined_features_unnormalized = np.vstack((initial_features_unnormalized, iteration_features_unnormalized))
            combined_labels = np.concatenate((initial_labels, iteration_labels))
            if "surface_H" in param_config.keys():
                combined_features_normalized_augmented = np.vstack((initial_features_normalized_augmented, iteration_features_normalized))
                combined_features_unnormalized_augmented = np.vstack((initial_features_unnormalized_augmented, iteration_features_unnormalized))
                combined_labels_augmented = np.concatenate((initial_labels_augmented, iteration_labels))
 
    # Save the combined data
    np.save(f"{training_data_path}/combined_features_normalized.npy", combined_features_normalized)
    np.save(f"{training_data_path}/combined_features_unnormalized.npy", combined_features_unnormalized)
    np.save(f"{training_data_path}/combined_labels.npy", combined_labels)

    if "surface_H" in param_config.keys():
        np.save(f"{training_data_path}/combined_features_normalized_augmented.npy", combined_features_normalized_augmented)
        np.save(f"{training_data_path}/combined_features_unnormalized_augmented.npy", combined_features_unnormalized_augmented)
        np.save(f"{training_data_path}/combined_labels_augmented.npy", combined_labels_augmented)

    ############################
    # Compiling stage4_outputs #
    ############################

    stage4_outputs = {}
    stage4_outputs["initial_features_normalized"] = initial_features_normalized
    stage4_outputs["initial_features_unnormalized"] = initial_features_unnormalized
    stage4_outputs["initial_labels"] = initial_labels

    if "surface_H" in param_config.keys():
        stage4_outputs["initial_features_normalized_augmented"] = initial_features_normalized_augmented
        stage4_outputs["initial_features_unnormalized_augmented"] = initial_features_unnormalized_augmented
        stage4_outputs["initial_labels_augmented"] = initial_labels_augmented

    if os.path.exists(f"{results_iter_common_path}/iteration_common/TDS_measurements.npy"):
        stage4_outputs["iteration_features_normalized"] = iteration_features_normalized
        stage4_outputs["iteration_features_unnormalized"] = iteration_features_unnormalized
        stage4_outputs["iteration_labels"] = iteration_labels

    
    stage4_outputs["combined_features_normalized"] = combined_features_normalized
    stage4_outputs["combined_features_unnormalized"] = combined_features_unnormalized
    stage4_outputs["combined_labels"] = combined_labels

    if "surface_H" in param_config.keys():
        stage4_outputs["combined_features_normalized_augmented"] = combined_features_normalized_augmented
        stage4_outputs["combined_features_unnormalized_augmented"] = combined_features_unnormalized_augmented
        stage4_outputs["combined_labels_augmented"] = combined_labels_augmented
    
    stage4_outputs["current_iteration_index"] = current_iteration_index
    
    return stage4_outputs

if __name__ == "__main__":
    global_configs = main_global_configs()
    stage2_outputs = main_prepare_common_data(global_configs)
    main_run_initial_sims(global_configs, stage2_outputs)
    main_prepare_sim_data(global_configs, stage2_outputs)

    