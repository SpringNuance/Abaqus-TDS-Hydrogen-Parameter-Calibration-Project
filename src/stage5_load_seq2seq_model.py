import numpy as np
import pandas as pd
import glob

from modules.initial_simulation import *
from utils.IO import *
from utils.calculation import *
from modules.stoploss import *

from optimization.LSTM import *
from optimization.transformer import *

from src.stage1_global_configs import * 
from src.stage2_prepare_common_data import *
from src.stage3_run_initial_sims import *
from src.stage4_prepare_initial_sim_data import *

def main_load_seq2seq_model(global_configs):

    # -----------------------------------------------------#
    #  Stage 5: Training the sequence to sequence ML model #
    # -----------------------------------------------------#

    all_paths = global_configs['all_paths']
    log_path = all_paths['log_path']
    models_path = all_paths['models_path']
    training_data_path = all_paths['training_data_path']
    targets_path = all_paths['targets_path']
    
    print_log("\n===========================================", log_path)
    print_log("= Stage 5: Load pretrained Seq2Seq models =", log_path)
    print_log("===========================================\n", log_path)

    # It is highly recommended to do this stage in the notebook file
    # on either CSC, Google Colab or Kaggle notebook to make use of GPU 
    # and visualize the process. 

    # This stage, therefore, only loads the pretrained models

    model_config = global_configs['model_config']
    
    transformer_hyperparams = model_config['transformer_hyperparams']
    transformer_model_name = transformer_hyperparams["model_name"]
    LSTM_hyperparams = model_config['LSTM_hyperparams']
    LSTM_model_name = LSTM_hyperparams["model_name"]

    use_referenced_flow_curve = model_config["use_referenced_flow_curve"]
    
    ## First, we need to check any latest models available in iteration folder
    # list all files in the models_path/iteration/iteration_{i} folder
    # They may exist if the user has run the optimization process
    # If they exist, we will load the latest model from the iteration folder with highest i 
    # If not, we have to load the initial model
    
    if use_referenced_flow_curve:
        print_log("We use the referenced flow curve to tell the scale of the flow curve", log_path)
        print_log("As a result, Transformer model is not needed in this case", log_path)
        print_log("We will only load the LSTM model", log_path)
        print_log("Verifying if the referenced flow curve is available", log_path)
        if not os.path.exists(f"{targets_path}/referenced_flow_curve.csv"):
            raise FileNotFoundError(f"The referenced flow curve file not found at '{targets_path}/referenced_flow_curve.csv'.\n\
                                    You must obtain it from tensile test of SDB geometry.")
        else:
            print_log(f"The referenced flow curve is found at '{targets_path}/referenced_flow_curve.csv'", log_path)
    else:
        print_log("We will use the Transformer model to predict the scale of the flow curve", log_path)
    
        #############################
        # LOADING TRANSFORMER MODEL #
        #############################
        
        model_dirs = glob.glob(f"{models_path}/transformer/iteration_*/{transformer_model_name}")
        
        # Extract the indices from the directory names
        indices = [int(os.path.basename(os.path.dirname(path)).split('_')[1]) for path in model_dirs]

        # Find the highest index or default to 0 if no directories are found
        index = max(indices, default=0)

        if index > 0:
            print_log(f"The latest Transformer model found in iteration_{index} folder", log_path)
            print_log(f"Loading the latest Transformer model from '{models_path}/transformer/iteration_{index}/{transformer_model_name}'", log_path)
            transformer_model_path = f"{models_path}/transformer/iteration_{index}/{transformer_model_name}"
        else:
            print_log(f"No latest Transformer model found in iteration folders. Loading the initial model", log_path)
            
            if not os.path.exists(f"{models_path}/transformer/initial/{transformer_model_name}"):
                raise FileNotFoundError(f"Pretrained Transformer model file not found at '{models_path}/transformer/initial/{transformer_model_name}'.\n\
                                        You must train the ML model first in the notebook folders.")
            else:
                print_log(f"The initial pretrained Transformer model found in '{models_path}/transformer/initial/{transformer_model_name}'", log_path)
                transformer_model_path = f"{models_path}/transformer/initial/{transformer_model_name}"
            

        initial_train_source_original_all = torch.load(f"{training_data_path}/initial_train_source_original_all.pt")
        initial_train_target_original_first = torch.load(f"{training_data_path}/initial_train_target_original_first.pt")

        # Parameters
        _, source_len, feature_size = initial_train_source_original_all.shape
        _, label_size, _ = initial_train_target_original_first.shape

        print_log(f"(Transformer) souce_len: {source_len}, feature_size: {feature_size}, label_size: {label_size}\n", log_path)

        d_model = transformer_hyperparams["d_model"]
        n_heads = transformer_hyperparams["n_heads"]
        num_layers = transformer_hyperparams["num_layers"]
        dim_feedforward = transformer_hyperparams["dim_feedforward"]
        activation_name = transformer_hyperparams["activation_name"]
        pos_enc_type = transformer_hyperparams["pos_enc_type"]
        encoder_layer_type = transformer_hyperparams["encoder_layer_type"]
        dropout = transformer_hyperparams["dropout"]

        # Load the model
        transformer_model = TransformerEncoder(feature_size, label_size, source_len,
                d_model, n_heads, num_layers, dim_feedforward, 
                activation_name, pos_enc_type, encoder_layer_type,
                dropout=dropout)

        try:
            transformer_model.load_state_dict(torch.load(transformer_model_path, map_location=torch.device('cpu')))
            print_log("The Transformer model has been loaded successfully\n", log_path)
        except:
            raise ValueError("The Transformer model could not be loaded. Please check the model file and the hyperparams.")

    # LSTM model is a must

    ######################
    # LOADING LSTM MODEL #
    ######################

    model_dirs = glob.glob(f"{models_path}/LSTM/iteration_*/{LSTM_model_name}")

    # Extract the indices from the directory names
    indices = [int(os.path.basename(os.path.dirname(path)).split('_')[1]) for path in model_dirs]

    # Find the highest index or default to 0 if no directories are found
    index = max(indices, default=0)

    if index > 0:
        print_log(f"The latest LSTM model found in iteration_{index} folder", log_path)
        print_log(f"Loading the latest LSTM model from '{models_path}/LSTM/iteration_{index}/{LSTM_model_name}'", log_path)
        LSTM_model_path = f"{models_path}/LSTM/iteration_{index}/{LSTM_model_name}"
    else:
        print_log(f"No latest LSTM model found in iteration folders. Loading the initial model", log_path)
          
        if not os.path.exists(f"{models_path}/LSTM/initial/{LSTM_model_name}"):
            raise FileNotFoundError(f"Pretrained LSTM model file not found at '{models_path}/LSTM/initial/{LSTM_model_name}'.\n\
                                    You must train the ML model first in the notebook folders.")
        else:
            print_log(f"Pretrained LSTM found at '{models_path}/LSTM/initial/{LSTM_model_name}'", log_path)
            LSTM_model_path = f"{models_path}/LSTM/initial/{LSTM_model_name}"
        
    initial_train_source_diff_all = torch.load(f"{training_data_path}/initial_train_source_diff_all.pt")
    initial_train_target_diff_last = torch.load(f"{training_data_path}/initial_train_target_diff_last.pt")

    # Parameters
    _, source_len, feature_size = initial_train_source_diff_all.shape
    _, target_len, label_size = initial_train_target_diff_last.shape
    
    print_log(f"(LSTM) feature_size: {feature_size}, label_size: {label_size}", log_path)
    print_log(f"(LSTM) source_len: {source_len}, target_len: {target_len}\n", log_path)

    hidden_size = LSTM_hyperparams["hidden_size"]
    num_layers = LSTM_hyperparams["num_layers"]
    dropout = LSTM_hyperparams["dropout"]
    bidirectional = LSTM_hyperparams["bidirectional"]
    use_attention = LSTM_hyperparams["use_attention"]
    attention_mechanism = LSTM_hyperparams["attention_mechanism"]

    # Load the model
    LSTM_model = LSTMModel(feature_size, label_size,
                source_len, target_len,
                hidden_size, num_layers,
                dropout=dropout,
                bidirectional=bidirectional, 
                use_attention=use_attention,
                attention_mechanism=attention_mechanism)
    try:
        LSTM_model.load_state_dict(torch.load(LSTM_model_path, map_location=torch.device('cpu')))
        print_log("The LSTM model has been loaded successfully", log_path)
    except:
        raise ValueError("The LSTM model could not be loaded. Please check the model file and the hyperparams.")
    
    if use_referenced_flow_curve:
        stage5_outputs = {
            "transformer_model": None,
            "LSTM_model": LSTM_model
        }
    else:
        stage5_outputs = {
            "transformer_model": transformer_model,
            "LSTM_model": LSTM_model
        }

    return stage5_outputs

if __name__ == "__main__":
    global_configs = main_global_configs()
    stage2_outputs = main_prepare_common_data(global_configs)
    main_run_initial_sims(global_configs, stage2_outputs)
    main_prepare_initial_sim_data(global_configs)
    main_load_seq2seq_model(global_configs)
    