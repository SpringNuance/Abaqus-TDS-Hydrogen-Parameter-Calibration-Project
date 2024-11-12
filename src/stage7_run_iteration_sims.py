import numpy as np
import glob

from modules.iteration_simulation import *
from modules.iteration_retraining import *
from modules.predict import *
from modules.stoploss import *

from utils.IO import *
from utils.calculation import *
from utils.hardening_laws import *
from utils.sampling import *

from src.stage1_global_configs import main_global_configs
from src.stage2_prepare_common_data import main_prepare_common_data
from src.stage3_run_initial_sims import main_run_initial_sims
from src.stage4_prepare_initial_sim_data import main_prepare_initial_sim_data
from src.stage5_load_seq2seq_model import main_load_seq2seq_model
from src.stage6_prepare_iteration_sim_data import main_prepare_iteration_sim_data

import time

def main_run_iteration_sims(global_configs, stage2_outputs, stage3_outputs, 
                            stage4_outputs, stage5_outputs, stage6_outputs):

    """
    Run iterative simulations integrating outputs from all previous stages.
    
    Parameters:
    - global_configs: Configuration settings used across all stages.
    - stage2_outputs: Outputs from common data preparation stage.
    - stage3_outputs: Outputs from running initial simulations
    - stage4_outputs: Outputs from initial simulation data preparation.
    - stage5_outputs: Outputs from training Seq2Seq models.
    - stage6_outputs: Outputs from iteration simulation data preparation.
    """

    # ------------------------------------#
    #  Stage 7: Run iterative simulations #
    # ------------------------------------#
    
    chosen_project_path = global_configs['chosen_project_path']
    project = global_configs['project']
    all_paths = global_configs['all_paths']
    models_path = all_paths['models_path']
    targets_path = all_paths['targets_path']
    objectives = global_configs['objectives']
    param_config = global_configs['param_config']
    param_config_inverse_fitting = global_configs['param_config_inverse_fitting']
    model_config = global_configs['model_config']
    iteration_sim_config = global_configs['iteration_sim_config']
    exp_yielding_index = global_configs['exp_yielding_index']
    stop_loss_config = global_configs['stop_loss_config']
    true_plastic_strain_config = global_configs['true_plastic_strain_config']
    
    # Extract from iteration_sim_config
    num_synthetic_predictions = iteration_sim_config['num_synthetic_predictions']
    sampling_method = iteration_sim_config['sampling_method']
    delete_sim = iteration_sim_config['delete_sim']
    iteration_array_job_config = iteration_sim_config['array_job_config']
    input_file_names = iteration_sim_config['input_file_names']
    
    # Extract from stop_loss_config
    stop_value_deviation_percent = stop_loss_config["stop_value_deviation_percent"]
    stop_num_points_percent = stop_loss_config["stop_num_points_percent"]
    loss_function = stop_loss_config["loss_function"]
    
    # Extract from model_config
    retrain_array_job_config = model_config['array_job_config']
    use_referenced_flow_curve = model_config["use_referenced_flow_curve"]
    scale_source = model_config["scale_source"]
    scale_target = model_config["scale_target"]
    divided_index = model_config["divided_index"]
    LSTM_model_name = model_config['LSTM_hyperparams']["model_name"]

    # Extract from true_plastic_strain_config
    hardening_law = true_plastic_strain_config["hardening_law"]
    extrapolate_N_first_strain_values =\
        true_plastic_strain_config["extrapolate_N_first_strain_values"] 
    
    log_path = all_paths['log_path']
    results_iter_common_path = all_paths['results_iter_common_path']

    #############################
    # Unpacking stage 2 outputs #
    #############################
    
    referenced_flow_curve_interpolated = stage2_outputs["referenced_flow_curve_interpolated"]
    true_plastic_strain = stage2_outputs["true_plastic_strain"]
    target_curves_interpolated_combined =\
        stage2_outputs["target_curves_interpolated_combined"]
    
    #############################
    # Unpacking stage 3 outputs #
    #############################

    initial_FD_curves_interpolated_combined =\
        stage3_outputs["initial_FD_curves_interpolated_combined"]
    
    #############################
    # Unpacking stage 4 outputs #
    #############################
    
    exp_source_original_all_unscaled = stage4_outputs["exp_source_original_all_unscaled"]
    exp_source_diff_all_unscaled = stage4_outputs["exp_source_diff_all_unscaled"]

    #############################
    # Unpacking stage 5 outputs #
    #############################
    
    if not use_referenced_flow_curve:
        transformer_model = stage5_outputs["transformer_model"]
        LSTM_model = stage5_outputs["LSTM_model"]
    else:
        transformer_model = None
        LSTM_model = stage5_outputs["LSTM_model"]
    
    #############################
    # Unpacking stage 6 outputs #
    #############################

    iteration_FD_curves_interpolated_combined =\
        stage6_outputs["iteration_FD_curves_interpolated_combined"]

    #######################
    # Stage 7 starts here #
    #######################

    print_log("\n======================================", log_path)
    print_log("= Stage 7: Run iterative simulations =", log_path)
    print_log("======================================\n", log_path)
    
    ######################################################
    # ITERATIVE OPTIMIZATION UNTIL STOP CONDITION IS MET #
    ######################################################

    while True:
        
        ##################################################################
        # CHECKING ITERATION INDEX AND WHETHER ANY ITERATION DATA EXISTS #
        ##################################################################

        model_dirs = glob.glob(f"{models_path}/LSTM/iteration_*/{LSTM_model_name}")

        iteration_index = len(model_dirs)

        if iteration_index == 0:
            iteration_data_exists = False
        else:
            iteration_data_exists = True    

        iteration_index += 1 

        #####################################################
        # CHECKING STOP DIAGNOSTIC                          #
        # Output: {log_path}/stop_diagnostic_initial.npy    #
        #         {log_path}/stop_diagnostic_iteration.npy  #
        #####################################################
        
        target_forces_interpolated_combined = {}
        target_displacements_interpolated_combined = {}
        initial_sim_forces_interpolated_combined = {}

        for objective in objectives:
            target_forces_interpolated_combined[objective] = target_curves_interpolated_combined[objective]['force']
            target_displacements_interpolated_combined[objective] = target_curves_interpolated_combined[objective]['displacement']
            initial_sim_forces_interpolated_combined[objective] = np.array([disp_force["force"] for params_tuple, disp_force\
                                                                            in initial_FD_curves_interpolated_combined[objective].items()])
            # print(target_forces_interpolated_combined[objective].shape) # (100,)
            # print(initial_sim_forces_interpolated_combined[objective].shape) # (256, 100)
        
        stop_diagnostic_initial = stop_condition_MOO(target_forces_interpolated_combined, 
                                                    target_displacements_interpolated_combined,
                                                    initial_sim_forces_interpolated_combined, 
                                                    objectives, stop_value_deviation_percent, 
                                                    stop_num_points_percent, loss_function)
        
        np.save(f"{log_path}/stop_diagnostic_initial.npy", stop_diagnostic_initial)

        print_log("\n########################################", log_path)
        print_log("# Stop diagnostic for initial simulation #", log_path)
        print_log("########################################\n", log_path)

        pretty_print_stop_diagnostic(stop_diagnostic_initial, objectives, log_path)

        if iteration_data_exists:

            target_forces_interpolated_combined = {}
            target_displacements_interpolated_combined = {}
            iteration_sim_forces_interpolated_combined = {}

            for objective in objectives:
                target_forces_interpolated_combined[objective] = target_curves_interpolated_combined[objective]['force']
                target_displacements_interpolated_combined[objective] = target_curves_interpolated_combined[objective]['displacement']
                iteration_sim_forces_interpolated_combined[objective] = np.array([disp_force["force"] for params_tuple, disp_force\
                                                                                  in iteration_FD_curves_interpolated_combined[objective].items()])
                # print(target_forces_interpolated_combined[objective].shape) # (100,)
                # print(iteration_sim_forces_interpolated_combined[objective].shape) # (number of iters, 100)
            
            stop_diagnostic_iteration = stop_condition_MOO(target_forces_interpolated_combined, 
                                                        target_displacements_interpolated_combined,
                                                        iteration_sim_forces_interpolated_combined, 
                                                        objectives, stop_value_deviation_percent, 
                                                        stop_num_points_percent, loss_function)
            
            np.save(f"{log_path}/stop_diagnostic_iteration.npy", stop_diagnostic_iteration)

            print_log("\n############################################", log_path)
            print_log("# Stop diagnostic for iteration simulation #", log_path)
            print_log("############################################\n", log_path)

            pretty_print_stop_diagnostic(stop_diagnostic_iteration, objectives, log_path)
        
        stop_condition_is_met = True
        stop_condition_is_met_initial = stop_diagnostic_initial["stop_condition_is_met_all_objectives_any_sim"]
        stop_condition_is_met &= stop_condition_is_met_initial
        
        if iteration_data_exists:
            stop_condition_is_met_iteration = stop_diagnostic_iteration["stop_condition_is_met_all_objectives_any_sim"]
            stop_condition_is_met &= stop_condition_is_met_iteration
        
        if stop_condition_is_met:
            print_log("\n Congratulations, the stop condition has been met.\nExiting the iteration simulation loop", log_path)
            break
        
        #############################################
        # PERFORMING PREDICTION FOR NEXT FLOW CURVE #
        # Output: exp_target_original_all           #
        #############################################

        if use_referenced_flow_curve:
            referenced_true_stress = referenced_flow_curve_interpolated['stress'][:(divided_index+1)]

            # Convert to tensor float
            referenced_exp_target_original_first = torch.tensor(referenced_true_stress, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

            print_log(f"The referenced exp target original first shape: {referenced_exp_target_original_first.shape}", log_path)
       
            exp_target_original_all\
                = seq2seq_predict_with_referenced_flow_curve(referenced_exp_target_original_first=referenced_exp_target_original_first, 
                                                        LSTM_model=LSTM_model,
                                                        exp_source_diff_all=exp_source_diff_all_unscaled,
                                                        scale_source=scale_source, scale_target=scale_target)
        else:
            exp_target_original_all\
                = seq2seq_predict_without_referenced_flow_curve(transformer_model=transformer_model, 
                                                                LSTM_model=LSTM_model, 
                                                                exp_source_original_all=exp_source_original_all_unscaled,
                                                                exp_source_diff_all=exp_source_diff_all_unscaled,
                                                                scale_source=scale_source, scale_target=scale_target)
        
        print_log(f"The predicted exp target original all shape: {exp_target_original_all.shape}", log_path)
        
        # Squeeze first and last dim and convert to np
        exp_target_original_all = exp_target_original_all.squeeze(0).squeeze(-1).detach().cpu().numpy()

        # assert that the flow true stress is monotonically increasing
        assert np.all(np.diff(exp_target_original_all) >= 0)
        
        ############################################
        # INVERSE CALCULATING THE NEXT PARAMETERS  #
        # Output: inverse_fitted_next_params_dict  #
        ############################################

        print_log("Start inverse calculating the next candidate parameters", log_path)
        inverse_fitted_next_params_dict, lowest_RMSE =\
            calculate_inverse_hardening_law_parameters(true_plastic_strain=true_plastic_strain[divided_index+1:], 
                                                        true_stress=exp_target_original_all[divided_index+1:], 
                                                        hardening_law=hardening_law, 
                                                        param_config=param_config_inverse_fitting, 
                                                        RMSE_threshold=100,
                                                        extrapolate_N_first_strain_values=0,
                                                        max_iter=100)
        
        print_log("\n" + 60 * "#" + "\n", log_path)
        print_log(f"Running iteration {iteration_index} for {project}" , log_path)
        print_log(f"The next inverse fitted parameters predicted by Seq2Seq models", log_path)
        pretty_print_parameters(inverse_fitted_next_params_dict, param_config, log_path)
        print_log(f"The fitted RMSE is {lowest_RMSE}", log_path)

        #############################################################################################################
        # SAMPLING SYNTHETIC PREDICTIONS                                                                            #
        # Output: {results_iter_common_path}/iteration_common/iteration_{iteration_index}_predicted_parameters.npy  #
        #         {results_iter_common_path}/iteration_common/iteration_{iteration_index}_predicted_true_stress.npy #
        #         {results_iter_common_path}/iteration_common/iteration_{iteration_index}_predicted_flow_curves.npy #
        #         {results_iter_common_path}/iteration_predicted_flow_curves.npy                                      #
        #         {results_iter_common_path}/iteration_predicted_parameters.npy                                       #
        #         {results_iter_common_path}/iteration_predicted_true_stress.npy                                      #
        #############################################################################################################

        # First, we append the true predicted flow curve to the data lists
        
        current_iteration_predicted_parameters = [inverse_fitted_next_params_dict]
        current_iteration_predicted_true_stress = [exp_target_original_all]       
        current_iteration_predicted_flow_curves = {tuple(inverse_fitted_next_params_dict.items()) : 
                                 {"strain": true_plastic_strain, "stress": exp_target_original_all}}

        # Start sampling num_synthetic_predictions based on this inverse_fitted_next_params_dict
        synthetic_samples = sampling_synthetic_predictions(param_config_inverse_fitting, inverse_fitted_next_params_dict, 
                                                            num_synthetic_predictions=num_synthetic_predictions * 10, 
                                                            method=sampling_method)

        # Then choose randomly num_synthetic_predictions samples from the synthetic_samples

        synthetic_samples = np.random.choice(synthetic_samples, num_synthetic_predictions, replace=False)

        for params_dict in synthetic_samples:
            current_iteration_predicted_parameters.append(params_dict)
            true_stress_sample = calculate_true_stress(params_dict, hardening_law,
                                                       true_plastic_strain, extrapolate_N_first_strain_values)

            current_iteration_predicted_true_stress.append(true_stress_sample)
            params_tuple = tuple(params_dict.items())
            current_iteration_predicted_flow_curves[params_tuple] = {"strain": true_plastic_strain, "stress": true_stress_sample}
        
        current_iteration_predicted_parameters = np.array(current_iteration_predicted_parameters)
        current_iteration_predicted_true_stress = np.array(current_iteration_predicted_true_stress)
        
        print_log(f"\nThere are {num_synthetic_predictions} synthetic flow curves generated", log_path)
        print_log(f"Together with true predicted flow curve, there are {len(current_iteration_predicted_parameters)} predictions", log_path)
        print_log(f"The iteration true stress shape is {current_iteration_predicted_true_stress.shape}", log_path)

        # Saving the parameters, true stress, and flow curves
        if os.path.exists(f"{results_iter_common_path}/iteration_common/iteration_{iteration_index}_predicted_parameters.npy"):
            print_log(f"Warning: The iteration {iteration_index} parameters, true stress, and flow curves already exist", log_path)
            print_log(f"We would not overwrite the existing files", log_path)
            current_iteration_predicted_parameters = np.load(f"{results_iter_common_path}/iteration_common/iteration_{iteration_index}_predicted_parameters.npy", allow_pickle=True)
            current_iteration_predicted_true_stress = np.load(f"{results_iter_common_path}/iteration_common/iteration_{iteration_index}_predicted_true_stress.npy", allow_pickle=True)
            current_iteration_predicted_flow_curves = np.load(f"{results_iter_common_path}/iteration_common/iteration_{iteration_index}_predicted_flow_curves.npy", allow_pickle=True)
        else:
            np.save(f"{results_iter_common_path}/iteration_common/iteration_{iteration_index}_predicted_parameters.npy", current_iteration_predicted_parameters)
            np.save(f"{results_iter_common_path}/iteration_common/iteration_{iteration_index}_predicted_true_stress.npy", current_iteration_predicted_true_stress)
            np.save(f"{results_iter_common_path}/iteration_common/iteration_{iteration_index}_predicted_flow_curves.npy", current_iteration_predicted_flow_curves)
        print_log(f"Saving iteration {iteration_index} parameters, true stress, and flow curves", log_path)
                
        ############################################################################################################################
        # RUNNING ITERATION SIMULATIONS FOR ALL OBJECTIVES                                                                         #
        # Output: {results_iter_common_path}/{objective}/FD_curves_iteration_{iteration_index}.npy for all objectives              #
        #         {results_iter_data}/{objective}/iteration_{iteration_index}/prediction_{prediction_index}/... for all objectives #
        ############################################################################################################################

        for objective in objectives: 

            # This iteration sim framework handles everything from preprocessing to postprocessing 
            iteration_sim_framework = IterationSimFramework(objective=objective,
                                                            delete_sim=delete_sim,
                                                            array_job_config=iteration_array_job_config,
                                                            all_paths=all_paths)
        
            if not os.path.exists(f"{results_iter_common_path}/{objective}/FD_curves_iteration_{iteration_index}.npy"):
                print_log("=======================================================================", log_path)
                print_log(f"(Iteration {iteration_index}) FD_curves.npy for {objective} does not exist", log_path)
            
                prediction_indices = range(1, num_synthetic_predictions + 2)
                input_file_name = input_file_names[objective]
                
                print_log("=======================================================================", log_path)
                print_log(f"(Iteration {iteration_index}) There is no FD_curves_iteration_{iteration_index}.npy for {objective}", log_path)
                print_log(f"(Iteration {iteration_index}) Program starts running the simulations of iteration {iteration_index} for {objective}", log_path)   
                print_log(f"(Iteration {iteration_index}) The prediction indices first and last values are {prediction_indices[0]} and {prediction_indices[-1]}", log_path)

                iteration_sim_framework.run_iteration_simulations(current_iteration_predicted_parameters, 
                                                                true_plastic_strain, current_iteration_predicted_true_stress, 
                                                                prediction_indices, iteration_index, input_file_name)
            
                print_log(f"Iteration {iteration_index} simulations for {objective} have completed", log_path)
    
            else: 
                print_log("=======================================================================", log_path)
                print_log(f"FD_curves_iteration_{iteration_index}.npy for {objective} already exists", log_path)
                
        ################################################################################################
        # DERIVING ITERATION SIM DATA                                                                  #
        # Output: {results_iter_common_path}/{objective}/FD_curves_interpolated.npy for all objectives #
        #         {results_iter_common_path}/FD_curves_interpolated_combined.npy                       #
        #         {training_data_path}/*.pth (all iteration training data)                             #
        ################################################################################################

        print_log("Start deriving iteration simulation data", log_path)

        stage6_outputs = main_prepare_iteration_sim_data(global_configs)

        iteration_FD_curves_interpolated_combined =\
            stage6_outputs["iteration_FD_curves_interpolated_combined"]

        print_log("Finish deriving iteration simulation data", log_path)

        time.sleep(180)

        #################################
        # RETRAINING THE SEQ2SEQ MODELS #
        #################################

        if not os.path.exists(f"{models_path}/LSTM/iteration_{iteration_index}/{LSTM_model_name}"):
            print_log("Start retraining the Seq2Seq models", log_path)

            iteration_retrain_framework = IterationRetrainFramework(chosen_project_path=chosen_project_path,
                                                                array_job_config=retrain_array_job_config,
                                                                all_paths=all_paths)
            
            iteration_retrain_framework.run_iteration_retraining(use_referenced_flow_curve=use_referenced_flow_curve,
                                                            current_iteration_index = iteration_index,
                                                            previous_iteration_index = iteration_index - 1,
                                                            )

            print_log("Finish retraining the Seq2Seq models", log_path)
        else:
            print_log(f"The trained models for iteration {iteration_index} already exist", log_path)
        
        #####################################
        # RELOADING THE NEW ITERATION MODEL #
        #####################################
        
        print_log("\nReloading the new iteration Seq2Seq models", log_path)

        stage5_outputs = main_load_seq2seq_model(global_configs)
        
        if not use_referenced_flow_curve:
            transformer_model = stage5_outputs["transformer_model"]
            LSTM_model = stage5_outputs["LSTM_model"]
        else:
            transformer_model = None
            LSTM_model = stage5_outputs["LSTM_model"]

if __name__ == "__main__":
    global_configs = main_global_configs()
    stage2_outputs = main_prepare_common_data(global_configs)
    stage3_outputs = main_run_initial_sims(global_configs, stage2_outputs)
    stage4_outputs = main_prepare_initial_sim_data(global_configs)
    stage5_outputs = main_load_seq2seq_model(global_configs)
    stage6_outputs = main_prepare_iteration_sim_data(global_configs)
    main_run_iteration_sims(global_configs, stage2_outputs, stage3_outputs,
                            stage4_outputs, stage5_outputs, stage6_outputs)