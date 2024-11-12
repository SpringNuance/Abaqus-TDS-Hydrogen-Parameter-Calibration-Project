import numpy as np
import time
from utils.IO import *
from utils.calculation import *
from modules.stoploss import *
from math import *
import os
import copy
import torch
import glob

from src.stage1_global_configs import * 
from src.stage2_prepare_common_data import *
from src.stage3_run_initial_sims import *
from src.stage4_prepare_initial_sim_data import *
from src.stage5_load_seq2seq_model import *

def main_prepare_iteration_sim_data(global_configs):

    # ---------------------------------------------- #
    #  Stage 6: Preparing iteration simulations data #
    # ---------------------------------------------- #
    
    all_paths = global_configs["all_paths"]
    objectives = global_configs["objectives"]
    
    results_iter_common_path = all_paths["results_iter_common_path"]
    targets_path = all_paths["targets_path"]
    training_data_path = all_paths["training_data_path"]
    log_path = all_paths["log_path"]
    models_path = all_paths["models_path"]

    print_log("\n==============================================", log_path)
    print_log("= Stage 6: Prepare iteration simulation data =", log_path)
    print_log("==============================================\n", log_path)
    
    # Iteration data exists only when FD_curves.npy exists for all objectives

    ##################################################################
    # CHECKING ITERATION INDEX AND WHETHER ANY ITERATION DATA EXISTS #
    ##################################################################
    
    interation_index_list = []
    for objective in objectives:
        iteration_index = len(glob.glob(f"{results_iter_common_path}/{objective}/FD_curves_iteration_*.npy"))
        interation_index_list.append(iteration_index)
    iteration_index = min(interation_index_list)

    if iteration_index == 0:
        iteration_data_exists = False
    else:
        iteration_data_exists = True    

    if not iteration_data_exists:
        print_log("There is no iteration data. This stage will do nothing", log_path)

        stage6_outputs = {

            "iteration_FD_curves_interpolated_combined": None,
            "iteration_predicted_parameters": None,
            "iteration_predicted_true_stress": None,
            "iteration_predicted_flow_curves": None,

            "iteration_source_original_all": None,
            "iteration_target_original_all": None,

            "iteration_train_source_original_all": None,
            "iteration_train_target_original_all": None,
            "iteration_test_source_original_all": None,
            "iteration_test_target_original_all": None,

            "iteration_train_source_diff_all": None,
            "iteration_train_source_diff_last": None,
            "iteration_train_target_diff_last": None,
            "iteration_train_target_original_first": None,

            "iteration_test_source_diff_all": None,
            "iteration_test_source_diff_last": None,
            "iteration_test_target_diff_last": None,
            "iteration_test_target_original_first": None,

        }
        return stage6_outputs
    else:

        ##################################
        # PROCESSING ITERATION SIMS DATA #
        ##################################

        # Now we combine the iterations of simulations together into FD_curves.npy
        print_log("=======================================================================", log_path)
        print_log(f"Start concatenating the iteration predictions together", log_path)

        for objective in objectives:
            FD_curves = {}

            for i in range(1, iteration_index + 1):
                FD_curves_iteration = np.load(f"{results_iter_common_path}/{objective}/FD_curves_iteration_{i}.npy", allow_pickle=True).tolist()
                FD_curves.update(FD_curves_iteration)
            
            np.save(f"{results_iter_common_path}/{objective}/FD_curves.npy", FD_curves)
            num_predictions = len(FD_curves)
            print_log(f"Finish concatenating the iterations together for {objective}", log_path)
            print_log(f"Number of iteration simulations for {objective}: {num_predictions} FD curves", log_path)


        # FD_curves_interpolated.npy for each objective
        # FD_curves_interpolated_combined for all

        from utils.calculation import interpolating_force

        FD_curves_interpolated_combined = {}

        for i, objective in enumerate(objectives):
 
            FD_curves_interpolated = {}
            FD_curves_iteration = np.load(f"{results_iter_common_path}/{objective}/FD_curves.npy", allow_pickle=True).tolist()
            interpolated_displacement_pd = pd.read_excel(f"{targets_path}/{objective}/interpolated_displacement.xlsx")           
            interpolated_displacement = interpolated_displacement_pd['displacement/m'].to_numpy()

            for params_tuple, FD_curve in FD_curves_iteration.items():
                sim_displacement = FD_curve['displacement']
                sim_force = FD_curve['force']
                interpolated_force = interpolating_force(sim_displacement, sim_force, interpolated_displacement)
                FD_curves_interpolated[params_tuple] = {'displacement': interpolated_displacement, 'force': interpolated_force}
            
            FD_curves_interpolated_combined[objective] = FD_curves_interpolated
            np.save(f"{results_iter_common_path}/{objective}/FD_curves_interpolated.npy", FD_curves_interpolated)
            print_log(f"Interpolated FD curves for {objective} are saved", log_path)

        np.save(f"{results_iter_common_path}/FD_curves_interpolated_combined.npy", FD_curves_interpolated_combined)
        print_log(f"Interpolated FD curves for all objectives are saved", log_path)


        # Now we combine all iterations together
        # iteration_predicted_parameters.npy
        # iteration_predicted_true_stress.npy
        # iteration_predicted_flow_curves.npy
        
        iteration_predicted_parameters = []
        iteration_predicted_true_stress = []
        iteration_predicted_flow_curves = {}

        for i in range(1, iteration_index + 1):
            iteration_predicted_parameters_i = np.load(f"{results_iter_common_path}/iteration_common/iteration_{i}_predicted_parameters.npy", allow_pickle=True)
            iteration_predicted_true_stress_i = np.load(f"{results_iter_common_path}/iteration_common/iteration_{i}_predicted_true_stress.npy", allow_pickle=True)
            iteration_predicted_flow_curves_i = np.load(f"{results_iter_common_path}/iteration_common/iteration_{i}_predicted_flow_curves.npy", allow_pickle=True).tolist()
            
            iteration_predicted_parameters.extend(iteration_predicted_parameters_i)
            iteration_predicted_true_stress.extend(iteration_predicted_true_stress_i)
            iteration_predicted_flow_curves.update(iteration_predicted_flow_curves_i)

        iteration_predicted_parameters = np.array(iteration_predicted_parameters)
        iteration_predicted_true_stress = np.array(iteration_predicted_true_stress)

        np.save(f"{results_iter_common_path}/iteration_predicted_flow_curves.npy", iteration_predicted_flow_curves)
        np.save(f"{results_iter_common_path}/iteration_predicted_parameters.npy", iteration_predicted_parameters)
        np.save(f"{results_iter_common_path}/iteration_predicted_true_stress.npy", iteration_predicted_true_stress)

        print_log(f"Saving combined iteration predicted parameters, true stress, and flow curves", log_path)

        print_log("Start running iteration simulation", log_path)

        ####################################
        # PREPARING TRAINING SEQUENCE DATA #
        ####################################
        
        # For all data below, 
        # source sequence is a multivariate FD curves on their respective interpolated displacements
        # target sequence is the univariate true stress on the true plastic strain

        # ------------------------------------------------#
        # Source sequence: iteration_source_original_all.pt #
        # ------------------------------------------------#

        results_iter_common_path = all_paths["results_iter_common_path"]
        training_data_path = all_paths["training_data_path"]
        targets_path = all_paths["targets_path"]

        model_config = global_configs["model_config"]

        interpolated_displacement_len = global_configs["interpolated_displacement_len"]
        print_log(f"\nThe interpolated displacement length is {interpolated_displacement_len}", log_path)
        
        iteration_source_original_all = []

        for i, objective in enumerate(objectives):
            source_sequence_one_sim = []
            FD_curves_interpolated = np.load(f"{results_iter_common_path}/{objective}/FD_curves_interpolated.npy", allow_pickle=True).tolist()
            for params_tuple, FD_curve_interpolated in FD_curves_interpolated.items():
                sim_force_interpolated = FD_curve_interpolated['force']
                source_sequence_one_sim.append(sim_force_interpolated)
            iteration_source_original_all.append(source_sequence_one_sim)

        iteration_source_original_all = np.array(iteration_source_original_all)
        iteration_source_original_all = torch.tensor(iteration_source_original_all)
        iteration_source_original_all = iteration_source_original_all.permute(1, 2, 0)

        print_log(f"\nThe source sequence is constructed with shape (num_sims, interpolated_displacement_len, num_objectives):", log_path)
        print_log(iteration_source_original_all.shape, log_path)
                    
        # Now we need to verify if this source sequence is constructed correctly

        for i, objective in enumerate(objectives):
            FD_curves_interpolated = np.load(f"{results_iter_common_path}/{objective}/FD_curves_interpolated.npy", allow_pickle=True).tolist()
            for sim_index, (params_tuple, FD_curve_interpolated) in enumerate(FD_curves_interpolated.items()):
                sim_force_interpolated = FD_curve_interpolated['force']
                sim_force_interpolated = torch.tensor(sim_force_interpolated)
                assert torch.allclose(iteration_source_original_all[sim_index, :, i], sim_force_interpolated)
            
        # --------------------------------------------------#
        # Target sequence: iteration_target_original_all.pt #
        # --------------------------------------------------#
        
        # target length totally depends on true_plastic_strain_config, we should not interpolate them
        
        iteration_target_original_all = []

        iteration_predicted_true_stress = np.load(f"{results_iter_common_path}/iteration_predicted_true_stress.npy", allow_pickle=True).tolist()
        iteration_target_original_all = torch.tensor(iteration_predicted_true_stress)

        # turn it into (num_sims, target_len, 1)
        iteration_target_original_all = iteration_target_original_all.unsqueeze(-1)
        print("\nThe target sequence is constructed with shape (num_sims, target_len, 1):")
        print(iteration_target_original_all.shape)

        # ---------------------------------------------------------------------------- #
        # Initial train data: iteration_train_source_original_all.pt (for Transformer) #
        #                     iteration_train_target_original_all.pt (probably unused) #
        # Initial test data:  iteration_test_source_original_all.pt  (for Transformer) #
        #                     iteration_test_target_original_all.pt  (probably unused) #
        # ---------------------------------------------------------------------------- #

        iteration_test_ratio = model_config["iteration_test_ratio"]

        # Now we split the source_sequence, target_sequence and exp_source_original_all into training and testing
        # There is no randomization
        
        num_sims = iteration_source_original_all.shape[0]

        num_test_sims = ceil(num_sims * iteration_test_ratio)
        num_train_sims = num_sims - num_test_sims

        iteration_train_source_original_all = iteration_source_original_all[:num_train_sims]
        iteration_train_target_original_all = iteration_target_original_all[:num_train_sims]
        iteration_test_source_original_all = iteration_source_original_all[num_train_sims:]
        iteration_test_target_original_all = iteration_target_original_all[num_train_sims:]

        print("\nThe training and testing source and target sequences are constructed with shapes:")
        print("iteration_train_source_original_all shape:", iteration_train_source_original_all.shape)
        print("iteration_train_target_original_all shape:", iteration_train_target_original_all.shape)
        print("iteration_test_source_original_all shape:", iteration_test_source_original_all.shape)
        print("iteration_test_target_original_all shape:", iteration_test_target_original_all.shape)

        # ----------------------------------------------------------------------------------- #
        # Initial train data: iteration_train_source_diff_all.pt  (for LSTM, preferred)       #
        #                     iteration_train_source_diff_last.pt (for LSTM, probably unused) #
        #                     iteration_train_target_diff_last.pt (for LSTM)                  #
        #                     iteration_train_target_original_first.pt (for Transformer)      #
        # Initial test data:  iteration_test_source_diff_all.pt  (for LSTM, preferred)        #
        #                     iteration_test_source_diff_last.pt (for LSTM, probably unused)  #
        #                     iteration_test_target_diff_last.pt (for LSTM)                   #
        #                     iteration_test_target_original_first.pt  (for Transformer)      #
        # ----------------------------------------------------------------------------------- #

        ##### However, due to the nature of the flow curve, which is always monotonically 
        # increasing, using the original flow curve as the target sequence would not be a good idea. 
        # Instead, we use the derivative of the flow curve as the target sequence. 
        # To be compatible, the FD curves are also differenced

        # The differentiation is simply done by subtracting the previous value from the current value, 
        # reducing the length of the sequence by 1. When training the Seq2Seq models, we can 
        # use a ReLU activation function to ensure the target sequence is always positive, 
        # reflecting the positive incremental flow curve

        # Question: how can we reconstruct the flow curves when we do not know the first value?
        # We would train two separate models, one to learn the incremental change, 
        # and one to solely learn the first N values. This N values is "divided_index" 
        # from model_config. LSTM is used to learn the incremental change, and Transformer
        # is used to learn the first N values.
        # However, we can use the referenced flow curve from SDB geometry and discard the need for Transformer
        # This can be set in configs as "use_referenced_flow_curve" as true

        divided_index = model_config["divided_index"]
        print_log(f"\nThe divided index is {divided_index}", log_path)
        
        iteration_train_source_diff_all = iteration_train_source_original_all[:, 1:, :] - iteration_train_source_original_all[:, :-1, :]
        iteration_train_source_diff_last = iteration_train_source_original_all[:, divided_index+1:, :] - iteration_train_source_original_all[:, divided_index:-1, :]
        iteration_train_target_diff_last = iteration_train_target_original_all[:, divided_index+1:, :] - iteration_train_target_original_all[:, divided_index:-1, :]
        iteration_train_target_original_first = iteration_train_target_original_all[:, :divided_index+1, :]

        iteration_test_source_diff_all = iteration_test_source_original_all[:, 1:, :] - iteration_test_source_original_all[:, :-1, :]
        iteration_test_source_diff_last = iteration_test_source_original_all[:, divided_index+1:, :] - iteration_test_source_original_all[:, divided_index:-1, :]
        iteration_test_target_diff_last = iteration_test_target_original_all[:, divided_index+1:, :] - iteration_test_target_original_all[:, divided_index:-1, :]
        iteration_test_target_original_first = iteration_test_target_original_all[:, :divided_index+1, :]

        print_log("\nThe training and testing source and target sequences are constructed with shapes:", log_path)
        print_log(f"iteration_train_source_diff_all shape: {str(iteration_train_source_diff_all.shape)}", log_path)
        print_log(f"iteration_train_source_diff_last shape: {str(iteration_train_source_diff_last.shape)}", log_path)
        print_log(f"iteration_train_target_diff_last shape: {str(iteration_train_target_diff_last.shape)}", log_path)
        print_log(f"iteration_train_target_original_first shape: {str(iteration_train_target_original_first.shape)}", log_path)
        print_log(f"iteration_test_source_diff_all shape: {str(iteration_test_source_diff_all.shape)}", log_path)
        print_log(f"iteration_test_source_diff_last shape: {str(iteration_test_source_diff_last.shape)}", log_path)
        print_log(f"iteration_test_target_diff_last shape: {str(iteration_test_target_diff_last.shape)}", log_path)
        print_log(f"iteration_test_target_original_first shape: {str(iteration_test_target_original_first.shape)}", log_path)
        
        # ---------------------------------------------------------------------#
        # The source and sequence target could be of vert different magnitudes #
        # Thus, we should scale them to a range to make the training easier    #
        # ---------------------------------------------------------------------#

        scale_source = model_config["scale_source"]
        scale_target = model_config["scale_target"]
        
        iteration_source_original_all = iteration_source_original_all * scale_source
        iteration_target_original_all = iteration_target_original_all * scale_target
        
        iteration_train_source_original_all = iteration_train_source_original_all * scale_source
        iteration_train_target_original_all = iteration_train_target_original_all * scale_target
        
        iteration_test_source_original_all = iteration_test_source_original_all * scale_source
        iteration_test_target_original_all = iteration_test_target_original_all * scale_target

        iteration_train_source_diff_all = iteration_train_source_diff_all * scale_source
        iteration_train_source_diff_last = iteration_train_source_diff_last * scale_source
        iteration_train_target_diff_last = iteration_train_target_diff_last * scale_target
        iteration_train_target_original_first = iteration_train_target_original_first * scale_target

        iteration_test_source_diff_all = iteration_test_source_diff_all * scale_source
        iteration_test_source_diff_last = iteration_test_source_diff_last * scale_source
        iteration_test_target_diff_last = iteration_test_target_diff_last * scale_target
        iteration_test_target_original_first = iteration_test_target_original_first * scale_target
        
        stage6_outputs = {

            "iteration_source_original_all": iteration_source_original_all,
            "iteration_target_original_all": iteration_target_original_all,

            "iteration_train_source_original_all": iteration_train_source_original_all,
            "iteration_train_target_original_all": iteration_train_target_original_all,
            "iteration_test_source_original_all": iteration_test_source_original_all,
            "iteration_test_target_original_all": iteration_test_target_original_all,

            "iteration_train_source_diff_all": iteration_train_source_diff_all,
            "iteration_train_source_diff_last": iteration_train_source_diff_last,
            "iteration_train_target_diff_last": iteration_train_target_diff_last,
            "iteration_train_target_original_first": iteration_train_target_original_first,

            "iteration_test_source_diff_all": iteration_test_source_diff_all,
            "iteration_test_source_diff_last": iteration_test_source_diff_last,
            "iteration_test_target_diff_last": iteration_test_target_diff_last,
            "iteration_test_target_original_first": iteration_test_target_original_first,

        }
        
        for training_data_name in stage6_outputs:
            torch.save(stage6_outputs[training_data_name], f"{training_data_path}/{training_data_name}.pt")
            print_log(f"{training_data_name} is saved", log_path)

        stage6_outputs["iteration_FD_curves_interpolated_combined"] = FD_curves_interpolated_combined
        stage6_outputs["iteration_predicted_parameters"] = iteration_predicted_parameters
        stage6_outputs["iteration_predicted_true_stress"] = iteration_predicted_true_stress
        stage6_outputs["iteration_predicted_flow_curves"] = iteration_predicted_flow_curves

        return stage6_outputs


if __name__ == "__main__":
    global_configs = main_global_configs()
    stage2_outputs = main_prepare_common_data(global_configs)
    main_run_initial_sims(global_configs, stage2_outputs)
    stage4_outputs = main_prepare_initial_sim_data(global_configs)
    stage5_outputs = main_load_seq2seq_model(global_configs)
    stage6_outputs = main_prepare_iteration_sim_data(global_configs)


    