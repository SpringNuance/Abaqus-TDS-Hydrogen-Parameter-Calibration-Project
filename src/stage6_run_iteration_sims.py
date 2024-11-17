import numpy as np
import glob

from modules.iteration_simulation import *
from modules.iteration_retraining import *
from modules.predict import *
from modules.stoploss import *

from optimization.GaussianProcess import *
from optimization.BayesianOptimization import *

from utils.IO import *
from utils.calculation import *
from utils.hardening_laws import *
from utils.sampling import *

from src.stage1_global_configs import main_global_configs
from src.stage2_prepare_common_data import main_prepare_common_data
from src.stage3_run_initial_sims import main_run_initial_sims
from src.stage4_prepare_sim_data import main_prepare_sim_data
from src.stage5_train_GP_model import main_train_GP_model

import time

def main_run_iteration_sims(global_configs, stage2_outputs, stage3_outputs, 
                            stage4_outputs, stage5_outputs):

    """
    Run iterative simulations integrating outputs from all previous stages.
    
    Parameters:
    - global_configs: Configuration settings used across all stages.
    - stage2_outputs: Outputs from common data preparation stage.
    - stage3_outputs: Outputs from running initial simulations
    - stage4_outputs: Outputs from initial simulation data preparation.
    - stage5_outputs: Outputs from training GP models.
    """

    # ------------------------------------#
    #  Stage 6: Run iterative simulations #
    # ------------------------------------#
    
    chosen_project_path = global_configs['chosen_project_path']
    project = global_configs['project']
    all_paths = global_configs['all_paths']
    models_path = all_paths['models_path']
    targets_path = all_paths['targets_path']
    param_config = global_configs['param_config']
    model_config = global_configs['model_config']
    optimization_config = global_configs['optimization_config']
    iteration_sim_config = global_configs['iteration_sim_config']
    stop_loss_config = global_configs['stop_loss_config']
    
    # Extract from iteration_sim_config
    delete_sim = iteration_sim_config['delete_sim']
    input_file_name = iteration_sim_config['input_file_name']
    
    # Extract from stop_loss_config
    stop_value_deviation_percent = stop_loss_config["stop_value_deviation_percent"]
    loss_function = stop_loss_config["loss_function"]
    
    # Extract from model_config
    model_name = model_config["model_name"]

    log_path = all_paths['log_path']
    results_iter_common_path = all_paths['results_iter_common_path']

    print_log("\n======================================", log_path)
    print_log("= Stage 6: Run iterative simulations =", log_path)
    print_log("======================================\n", log_path)

    #############################
    # Unpacking stage 2 outputs #
    #############################

    initial_sampled_parameters = stage2_outputs["initial_sampled_parameters"]
    target_TDS_measurements = stage2_outputs["target_TDS_measurements"]
    description_properties_dict = stage2_outputs["description_properties_dict"]
    
    #############################
    # Unpacking stage 3 outputs #
    #############################

    initial_sim_TDS_measurements = stage3_outputs["initial_sim_TDS_measurements"]
    
    #############################
    # Unpacking stage 4 outputs #
    #############################
    
    current_iteration_index = stage4_outputs["current_iteration_index"]
    
    if current_iteration_index > 0:
        initial_sim_TDS_measurements = stage3_outputs["initial_sim_TDS_measurements"]

    combined_features_normalized = stage4_outputs["combined_features_normalized"]
    combined_features_unnormalized = stage4_outputs["combined_features_unnormalized"]
    combined_labels = stage4_outputs["combined_labels"]

    if "surface_H" in param_config.keys():
        combined_features_normalized_augmented = stage4_outputs["combined_features_normalized_augmented"]
        combined_features_unnormalized_augmented = stage4_outputs["combined_features_unnormalized_augmented"]
        combined_labels_augmented = stage4_outputs["combined_labels_augmented"]

    #############################
    # Unpacking stage 5 outputs #
    #############################
    
    GP_model_wrapper = stage5_outputs["GP_model_wrapper"]
    GP_model = GP_model_wrapper.GP_model
    
    #############################################
    # PERFORMING PREDICTION FOR NEXT PARAMETERS #
    #############################################

    train_Y = combined_labels if "surface_H" not in param_config.keys() else combined_labels_augmented
    num_params = len(param_config)
    bounds = torch.tensor([[0.0] * num_params, [1.0] * num_params], dtype=dtype, device=device)
    BayesOptWrapper = BayesianOptimizationWrapper(all_paths, GP_model, train_Y, 
                 bounds, param_config, optimization_config)
    normalized_candidates, acq_values = BayesOptWrapper.optimize_acq_function()
    deformalized_candidates = denormalize_points(normalized_candidates, param_config)
    print_log("\n" + 60 * "#" + "\n", log_path)
    print_log(f"Bayesian Optimization has found {len(deformalized_candidates)} candidates", log_path)
    print_log(f"The acquisition values are {acq_values}", log_path)

    # Now convert candidates to a list of dictionaries
    next_iteration_predicted_parameters = []
    for i, candidate in enumerate(deformalized_candidates):
        next_iteration_predicted_parameters.append({key: value for key, value in zip(param_config.keys(), candidate)})
    pretty_print_parameters(next_iteration_predicted_parameters, param_config, log_path)
    
    # Saving the parameters

    next_iteration_index = current_iteration_index + 1
    if not os.path.exists(f"{results_iter_common_path}/iteration_{next_iteration_index}"):
        os.makedirs(f"{results_iter_common_path}/iteration_{next_iteration_index}")

    np.save(f"{results_iter_common_path}/iteration_{next_iteration_index}/predicted_parameters.npy", next_iteration_predicted_parameters)
    print_log(f"Saving iteration {next_iteration_index} parameters", log_path)
    
    #####################################################
    # RUNNING ITERATION SIMULATIONS FOR ALL OBJECTIVES  #  
    #####################################################

    # This iteration sim framework handles everything from preprocessing to postprocessing 
    iteration_sim_framework = IterationSimFramework(param_config, description_properties_dict, 
                                                    delete_sim=delete_sim,
                                                    all_paths=all_paths) 

    if not os.path.exists(f"{results_iter_common_path}/iteration_{next_iteration_index}/TDS_measurements.npy"):

        prediction_indices = range(1, len(next_iteration_predicted_parameters) + 1)
        
        print_log("=======================================================================", log_path)
        print_log(f"(Iteration {next_iteration_index}) There is no TDS_measurements.npy for iteration {next_iteration_index}", log_path)
        print_log(f"(Iteration {next_iteration_index}) Program starts running the simulations of iteration {next_iteration_index}", log_path)   
        print_log(f"(Iteration {next_iteration_index}) The prediction indices first and last values are {prediction_indices[0]} and {prediction_indices[-1]}", log_path)

        iteration_sim_framework.run_iteration_simulations(next_iteration_predicted_parameters, 
                                                        prediction_indices, next_iteration_index, input_file_name)

        print_log(f"Iteration {next_iteration_index} simulations have completed", log_path)

    else: 
        print_log("=======================================================================", log_path)
        print_log(f"TDS_measurements_iteration_{next_iteration_index}.npy already exists", log_path)

if __name__ == "__main__":
    global_configs = main_global_configs()
    stage2_outputs = main_prepare_common_data(global_configs)
    stage3_outputs = main_run_initial_sims(global_configs, stage2_outputs)
    stage4_outputs = main_prepare_sim_data(global_configs, stage2_outputs)
    stage5_outputs = main_train_GP_model(global_configs, stage4_outputs)
    main_run_iteration_sims(global_configs, stage2_outputs, stage3_outputs,
                            stage4_outputs, stage5_outputs)