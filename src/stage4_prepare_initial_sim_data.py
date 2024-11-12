import numpy as np
import time
from utils.IO import *
from utils.calculation import *
from modules.stoploss import *
from math import *
import os
import copy
import torch

from src.stage1_global_configs import * 
from src.stage2_prepare_common_data import *
from src.stage3_run_initial_sims import *

def main_prepare_initial_sim_data(global_configs):

    # --------------------------------------------- #
    #  Stage 4: Preparing initial simulations data  #
    # --------------------------------------------- #
    
    all_paths = global_configs["all_paths"]
    objectives = global_configs["objectives"]
    
    results_init_common_path = all_paths["results_init_common_path"]
    targets_path = all_paths["targets_path"]
    training_data_path = all_paths["training_data_path"]
    log_path = all_paths["log_path"]

    print_log("\n============================================", log_path)
    print_log("= Stage 4: Prepare initial simulation data =", log_path)
    print_log("============================================\n", log_path)

    ################################
    # PROCESSING INITIAL SIMS DATA #
    ################################
        
    ####################################
    # PREPARING TRAINING SEQUENCE DATA #
    ####################################
    
    # For all data below, 
    # source sequence is a multivariate FD curves on their respective interpolated displacements
    # target sequence is the univariate true stress on the true plastic strain

    # ------------------------------------------------#
    # Source sequence: initial_source_original_all.pt #
    # ------------------------------------------------#

    results_init_common_path = all_paths["results_init_common_path"]
    training_data_path = all_paths["training_data_path"]
    targets_path = all_paths["targets_path"]

    model_config = global_configs["model_config"]

    interpolated_displacement_len = global_configs["interpolated_displacement_len"]
    print_log(f"\nThe interpolated displacement length is {interpolated_displacement_len}", log_path)
    
    initial_source_original_all = []

    for i, objective in enumerate(objectives):
        source_sequence_one_sim = []
        if os.path.exists(f"{results_init_common_path}/{objective}/FD_curves_interpolated.npy"):
            # Plotting the mean simulated FD curve
            FD_curves_interpolated = np.load(f"{results_init_common_path}/{objective}/FD_curves_interpolated.npy", allow_pickle=True).tolist()
            for params_tuple, FD_curve_interpolated in FD_curves_interpolated.items():
                sim_force_interpolated = FD_curve_interpolated['force']
                source_sequence_one_sim.append(sim_force_interpolated)
            initial_source_original_all.append(source_sequence_one_sim)

    initial_source_original_all = np.array(initial_source_original_all)
    # Convert to tensor
    initial_source_original_all = torch.tensor(initial_source_original_all)
    initial_source_original_all = initial_source_original_all.permute(1, 2, 0)

    print_log(f"\nThe source sequence is constructed with shape (num_sims, interpolated_displacement_len, num_objectives):", log_path)
    print_log(initial_source_original_all.shape, log_path)
                
    # Now we need to verify if this source sequence is constructed correctly

    for i, objective in enumerate(objectives):
        if os.path.exists(f"{results_init_common_path}/{objective}/FD_curves_interpolated.npy"):
            FD_curves_interpolated = np.load(f"{results_init_common_path}/{objective}/FD_curves_interpolated.npy", allow_pickle=True).tolist()
            for sim_index, (params_tuple, FD_curve_interpolated) in enumerate(FD_curves_interpolated.items()):
                sim_force_interpolated = FD_curve_interpolated['force']
                sim_force_interpolated = torch.tensor(sim_force_interpolated)
                assert torch.allclose(initial_source_original_all[sim_index, :, i], sim_force_interpolated)
        
    # ------------------------------------------------#
    # Target sequence: initial_target_original_all.pt #
    # ------------------------------------------------#
    
    # target length totally depends on true_plastic_strain_config, we should not interpolate them
    
    initial_target_original_all = []

    if os.path.exists(f"{results_init_common_path}/initial_sampled_true_stress.npy"):

        initial_sampled_true_stress = np.load(f"{results_init_common_path}/initial_sampled_true_stress.npy", allow_pickle=True).tolist()
        initial_target_original_all = torch.tensor(initial_sampled_true_stress)

    # turn it into (num_sims, target_len, 1)
    initial_target_original_all = initial_target_original_all.unsqueeze(-1)
    print("\nThe target sequence is constructed with shape (num_sims, target_len, 1):")
    print(initial_target_original_all.shape)

    # -------------------------------------------------------------------------- #
    # Initial train data: initial_train_source_original_all.pt (for Transformer) #
    #                     initial_train_target_original_all.pt (probably unused) #
    # Initial test data:  initial_test_source_original_all.pt  (for Transformer) #
    #                     initial_test_target_original_all.pt  (probably unused) #
    # -------------------------------------------------------------------------- #

    initial_test_ratio = model_config["initial_test_ratio"]

    # Now we split the source_sequence, target_sequence and exp_source_original_all into training and testing
    # There is no randomization
    
    num_sims = initial_source_original_all.shape[0]

    num_test_sims = ceil(num_sims * initial_test_ratio)
    num_train_sims = num_sims - num_test_sims

    initial_train_source_original_all = initial_source_original_all[:num_train_sims]
    initial_train_target_original_all = initial_target_original_all[:num_train_sims]
    initial_test_source_original_all = initial_source_original_all[num_train_sims:]
    initial_test_target_original_all = initial_target_original_all[num_train_sims:]

    print("\nThe training and testing source and target sequences are constructed with shapes:")
    print("initial_train_source_original_all shape:", initial_train_source_original_all.shape)
    print("initial_train_target_original_all shape:", initial_train_target_original_all.shape)
    print("initial_test_source_original_all shape:", initial_test_source_original_all.shape)
    print("initial_test_target_original_all shape:", initial_test_target_original_all.shape)

    # --------------------------------------------------------------------------------- #
    # Initial train data: initial_train_source_diff_all.pt  (for LSTM, preferred)       #
    #                     initial_train_source_diff_last.pt (for LSTM, probably unused) #
    #                     initial_train_target_diff_last.pt (for LSTM)                  #
    #                     initial_train_target_original_first.pt (for Transformer)      #
    # Initial test data:  initial_test_source_diff_all.pt  (for LSTM, preferred)        #
    #                     initial_test_source_diff_last.pt (for LSTM, probably unused)  #
    #                     initial_test_target_diff_last.pt (for LSTM)                   #
    #                     initial_test_target_original_first.pt  (for Transformer)      #
    # --------------------------------------------------------------------------------- #

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
    
    initial_train_source_diff_all = initial_train_source_original_all[:, 1:, :] - initial_train_source_original_all[:, :-1, :]
    initial_train_source_diff_last = initial_train_source_original_all[:, divided_index+1:, :] - initial_train_source_original_all[:, divided_index:-1, :]
    initial_train_target_diff_last = initial_train_target_original_all[:, divided_index+1:, :] - initial_train_target_original_all[:, divided_index:-1, :]
    initial_train_target_original_first = initial_train_target_original_all[:, :divided_index+1, :]
    

    initial_test_source_diff_all = initial_test_source_original_all[:, 1:, :] - initial_test_source_original_all[:, :-1, :]
    initial_test_source_diff_last = initial_test_source_original_all[:, divided_index+1:, :] - initial_test_source_original_all[:, divided_index:-1, :]
    initial_test_target_diff_last = initial_test_target_original_all[:, divided_index+1:, :] - initial_test_target_original_all[:, divided_index:-1, :]
    initial_test_target_original_first = initial_test_target_original_all[:, :divided_index+1, :]

    print_log("\nThe training and testing source and target sequences are constructed with shapes:", log_path)
    print_log(f"initial_train_source_diff_all shape: {str(initial_train_source_diff_all.shape)}", log_path)
    print_log(f"initial_train_source_diff_last shape: {str(initial_train_source_diff_last.shape)}", log_path)
    print_log(f"initial_train_target_diff_last shape: {str(initial_train_target_diff_last.shape)}", log_path)
    print_log(f"initial_train_target_original_first shape: {str(initial_train_target_original_first.shape)}", log_path)
    print_log(f"initial_test_source_diff_all shape: {str(initial_test_source_diff_all.shape)}", log_path)
    print_log(f"initial_test_source_diff_last shape: {str(initial_test_source_diff_last.shape)}", log_path)
    print_log(f"initial_test_target_diff_last shape: {str(initial_test_target_diff_last.shape)}", log_path)
    print_log(f"initial_test_target_original_first shape: {str(initial_test_target_original_first.shape)}", log_path)
    
    # ------------------------------------------------#
    # Exp source sequence: exp_source_original_all.pt #
    # ------------------------------------------------#

    # Now we construct the exp_source_original_all, which is used to predict flow curve
    # There is only 1 experimental data so it should have shape 
    # (1, interpolated_displacement_len, num_objectives)

    targets_path = all_paths["targets_path"]
    exp_source_original_all = []

    for i, objective in enumerate(objectives):
        FD_curve_final_interpolated = pd.read_excel(f"{targets_path}/{objective}/FD_curve_final_interpolated.xlsx", engine='openpyxl')
        exp_force_interpolated = FD_curve_final_interpolated['force/N'].values
        exp_source_original_all.append(exp_force_interpolated)

    exp_source_original_all = np.array(exp_source_original_all)
    # Convert to tensor
    exp_source_original_all = torch.tensor(exp_source_original_all).permute(1,0)
    exp_source_original_all = exp_source_original_all.unsqueeze(0)

    print_log("\nThe exp_source_original_all is constructed with shape (1, interpolated_displacement_len, num_objectives):", log_path)
    print_log(exp_source_original_all.shape, log_path)
    

    # Verify if this exp_source_original_all is constructed correctly
    for i, objective in enumerate(objectives):
        FD_curve_final_interpolated = pd.read_excel(f"{targets_path}/{objective}/FD_curve_final_interpolated.xlsx", engine='openpyxl')
        exp_force_interpolated = FD_curve_final_interpolated['force/N'].values
        exp_force_interpolated = torch.tensor(exp_force_interpolated)
        assert torch.allclose(exp_source_original_all[0, :, i], exp_force_interpolated)
    
    # Difference the exp_source_original_all
    exp_source_diff_all = exp_source_original_all[:, 1:, :] - exp_source_original_all[:, :-1, :]
    print_log("The exp_source_diff_all is constructed with shape (1, interpolated_displacement_len-1, num_objectives):", log_path)
    print_log(exp_source_diff_all.shape, log_path)
    print_log("\n", log_path)

    # ---------------------------------------------------------------------#
    # The source and sequence target could be of vert different magnitudes #
    # Thus, we should scale them to a range to make the training easier    #
    # ---------------------------------------------------------------------#

    scale_source = model_config["scale_source"]
    scale_target = model_config["scale_target"]
    
    initial_source_original_all = initial_source_original_all * scale_source
    initial_target_original_all = initial_target_original_all * scale_target
    
    initial_train_source_original_all = initial_train_source_original_all * scale_source
    initial_train_target_original_all = initial_train_target_original_all * scale_target
    
    initial_test_source_original_all = initial_test_source_original_all * scale_source
    initial_test_target_original_all = initial_test_target_original_all * scale_target

    initial_train_source_diff_all = initial_train_source_diff_all * scale_source
    initial_train_source_diff_last = initial_train_source_diff_last * scale_source
    initial_train_target_diff_last = initial_train_target_diff_last * scale_target
    initial_train_target_original_first = initial_train_target_original_first * scale_target

    initial_test_source_diff_all = initial_test_source_diff_all * scale_source
    initial_test_source_diff_last = initial_test_source_diff_last * scale_source
    initial_test_target_diff_last = initial_test_target_diff_last * scale_target
    initial_test_target_original_first = initial_test_target_original_first * scale_target

    exp_source_original_all_scaled = exp_source_original_all * scale_source
    exp_source_diff_all_scaled = exp_source_diff_all * scale_source

    stage4_outputs = {

        "initial_source_original_all": initial_source_original_all,
        "initial_target_original_all": initial_target_original_all,

        "initial_train_source_original_all": initial_train_source_original_all,
        "initial_train_target_original_all": initial_train_target_original_all,
        "initial_test_source_original_all": initial_test_source_original_all,
        "initial_test_target_original_all": initial_test_target_original_all,

        "initial_train_source_diff_all": initial_train_source_diff_all,
        "initial_train_source_diff_last": initial_train_source_diff_last,
        "initial_train_target_diff_last": initial_train_target_diff_last,
        "initial_train_target_original_first": initial_train_target_original_first,

        "initial_test_source_diff_all": initial_test_source_diff_all,
        "initial_test_source_diff_last": initial_test_source_diff_last,
        "initial_test_target_diff_last": initial_test_target_diff_last,
        "initial_test_target_original_first": initial_test_target_original_first,

        "exp_source_original_all_scaled": exp_source_original_all_scaled, 
        "exp_source_diff_all_scaled": exp_source_diff_all_scaled,

        "exp_source_original_all_unscaled": exp_source_original_all,
        "exp_source_diff_all_unscaled": exp_source_diff_all,
    }
    
    for training_data_name in stage4_outputs:
        if not os.path.exists(f"{training_data_path}/{training_data_name}.pt"):
            torch.save(stage4_outputs[training_data_name], f"{training_data_path}/{training_data_name}.pt")
            print_log(f"{training_data_name} is saved", log_path)
        else:
            training_data = torch.load(f"{training_data_path}/{training_data_name}.pt")
            print_log(f"{training_data_name} already exists", log_path)
            stage4_outputs[training_data_name] = training_data

    return stage4_outputs


if __name__ == "__main__":
    global_configs = main_global_configs()
    stage2_outputs = main_prepare_common_data(global_configs)
    main_run_initial_sims(global_configs, stage2_outputs)
    main_prepare_initial_sim_data(global_configs)

    