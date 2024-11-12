
import numpy as np
import os
import torch
import argparse
import json

from optimization.transformer_helper import *
from optimization.transformer import *
from src.stage0_initialize_directory import *
from utils.IO import *

def retrain_model(chosen_project_path, current_iteration_index, previous_iteration_index):
    
    with open(chosen_project_path) as f:
        global_configs = json.load(f)

    project = global_configs['project']
    objectives = global_configs['objectives']

    all_paths = initialize_directory(project, objectives)

    training_data_path = all_paths['training_data_path']
    models_path = all_paths['models_path']
    log_path = all_paths['log_path']
    
    training_log_file_name = f"iteration_{current_iteration_index}_retrain.log"
    
    if os.path.exists(f"{log_path}/{training_log_file_name}"):
        os.remove(f"{log_path}/{training_log_file_name}")
    
    with open(f"{log_path}/{training_log_file_name}", 'w') as log_file:
        log_file.write("")

    print_log("The current directory is: ", log_path = log_path, file_name = training_log_file_name)
    print_log(os.getcwd() + "\n", log_path = log_path, file_name = training_log_file_name)
    # Example of chosen_project_path: "configs/global_config_CP1000_RD_20C.json"
    print_log(f"The chosen project path is: {chosen_project_path}", log_path = log_path, file_name = training_log_file_name)

    # Loading training and testing initial data

    initial_train_source_original_all = torch.load(f"{training_data_path}/initial_train_source_original_all.pt")
    initial_train_target_original_first = torch.load(f"{training_data_path}/initial_train_target_original_first.pt")
    initial_test_source_original_all = torch.load(f"{training_data_path}/initial_test_source_original_all.pt")
    initial_test_target_original_first = torch.load(f"{training_data_path}/initial_test_target_original_first.pt")

    # Convert them to float32

    initial_train_source_original_all = initial_train_source_original_all.float()
    initial_train_target_original_first = initial_train_target_original_first.float()
    initial_test_source_original_all = initial_test_source_original_all.float()
    initial_test_target_original_first = initial_test_target_original_first.float()

    print_log(f"\nShape of the initial_train_source_original_all: {initial_train_source_original_all.shape}", log_path = log_path, file_name = training_log_file_name)
    print_log(f"Shape of the initial_train_target_original_first: {initial_train_target_original_first.shape}", log_path = log_path, file_name = training_log_file_name)
    print_log(f"Shape of the initial_test_source_original_all: {initial_test_source_original_all.shape}", log_path = log_path, file_name = training_log_file_name)
    print_log(f"Shape of the initial_test_target_original_first: {initial_test_target_original_first.shape}", log_path = log_path, file_name = training_log_file_name)

    # Check if any of them has NaN or infinite values

    print_log(f"\nNumber of NaN values in initial_train_source_original_all: {np.isnan(initial_train_source_original_all).sum()}", log_path = log_path, file_name = training_log_file_name)
    print_log(f"Number of NaN values in initial_train_target_original_first: {np.isnan(initial_train_target_original_first).sum()}", log_path = log_path, file_name = training_log_file_name)
    print_log(f"Number of NaN values in initial_test_source_original_all: {np.isnan(initial_test_source_original_all).sum()}", log_path = log_path, file_name = training_log_file_name)
    print_log(f"Number of NaN values in initial_test_target_original_first: {np.isnan(initial_test_target_original_first).sum()}", log_path = log_path, file_name = training_log_file_name)

    print_log(f"\nNumber of infinite values in initial_train_source_original_all: {np.isinf(initial_train_source_original_all).sum()}", log_path = log_path, file_name = training_log_file_name)
    print_log(f"Number of infinite values in initial_train_target_original_first: {np.isinf(initial_train_target_original_first).sum()}", log_path = log_path, file_name = training_log_file_name)
    print_log(f"Number of infinite values in initial_test_source_original_all: {np.isinf(initial_test_source_original_all).sum()}", log_path = log_path, file_name = training_log_file_name)
    print_log(f"Number of infinite values in initial_test_target_original_first: {np.isinf(initial_test_target_original_first).sum()}", log_path = log_path, file_name = training_log_file_name)

    # Ensure that all target_sequence are positive
    print_log(f"\nNumber of values <= 0 in initial_train_target_original_first: {(initial_train_target_original_first <= 0).sum()}", log_path = log_path, file_name = training_log_file_name)
    print_log(f"Number of values <= 0 in initial_test_target_original_first: {(initial_test_target_original_first <= 0).sum()}", log_path = log_path, file_name = training_log_file_name)
    # Ensure that the scale of the source sequence is correct
    print_log("The first time step of the initial_train_source_original_all tensor is:", log_path = log_path, file_name = training_log_file_name)
    print_log(initial_train_source_original_all[0][0], log_path = log_path, file_name = training_log_file_name)

    # Ensure that the scale of the target sequence is correct
    print_log("The first time step of the initial_train_target_original_first tensor is:", log_path = log_path, file_name = training_log_file_name)
    print_log(initial_train_target_original_first[0][0], log_path = log_path, file_name = training_log_file_name)

    # Loading training iteration data

    iteration_train_source_sequence = torch.load(f"{training_data_path}/iteration_train_source_sequence.pt") 
    iteration_train_target_sequence_first = torch.load(f"{training_data_path}/iteration_train_target_sequence_first.pt")

    iteration_train_source_sequence = iteration_train_source_sequence.float()
    iteration_train_target_sequence_first = iteration_train_target_sequence_first.float()

    print_log(f"\nShape of the iteration_train_source_sequence: {iteration_train_source_sequence.shape}", log_path = log_path, file_name = training_log_file_name)
    print_log(f"Shape of the iteration_train_target_sequence_first: {iteration_train_target_sequence_first.shape}", log_path = log_path, file_name = training_log_file_name)

    print_log(f"Number of NaN values in iteration_train_source_sequence: {np.isnan(iteration_train_source_sequence).sum()}", log_path = log_path, file_name = training_log_file_name)
    print_log(f"Number of NaN values in iteration_train_target_sequence_first: {np.isnan(iteration_train_target_sequence_first).sum()}", log_path = log_path, file_name = training_log_file_name)

    print_log(f"Number of infinite values in iteration_train_source_sequence: {np.isinf(iteration_train_source_sequence).sum()}", log_path = log_path, file_name = training_log_file_name)
    print_log(f"Number of infinite values in iteration_train_target_sequence_first: {np.isinf(iteration_train_target_sequence_first).sum()}", log_path = log_path, file_name = training_log_file_name)

    print_log(f"Number of values <= in iteration_train_target_sequence_first: {(iteration_train_target_sequence_first < 0).sum()}", log_path = log_path, file_name = training_log_file_name)

    print_log("The first time step of the iteration_train_source_sequence tensor is:", log_path = log_path, file_name = training_log_file_name)
    print_log(iteration_train_source_sequence[0][0], log_path = log_path, file_name = training_log_file_name)

    print_log("The first time step of the iteration_train_target_sequence_first tensor is:", log_path = log_path, file_name = training_log_file_name)
    print_log(iteration_train_target_sequence_first[0][0], log_path = log_path, file_name = training_log_file_name)

    # Now we would like to concatenate the initial and iteration training data

    combined_train_source_original_all = torch.cat((initial_train_source_original_all, iteration_train_source_sequence), dim=0)
    combined_train_target_original_first = torch.cat((initial_train_target_original_first, iteration_train_target_sequence_first), dim=0)
    
    print_log(f"\nShape of the combined_train_source_original_all: {combined_train_source_original_all.shape}", log_path = log_path, file_name = training_log_file_name)
    print_log(f"Shape of the combined_train_target_original_first: {combined_train_target_original_first.shape}", log_path = log_path, file_name = training_log_file_name)

    # Obtain the device

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_log(f"The training device: {device}", log_path = log_path, file_name = training_log_file_name)

    ###############################
    # Loading the hyperparameters #
    ###############################

    model_config = global_configs['model_config']
    
    transformer_hyperparams = model_config['transformer_hyperparams']
    transformer_retraining = model_config['transformer_retraining']
    
    model_name = transformer_hyperparams["model_name"]
    d_model = transformer_hyperparams["d_model"]
    n_heads = transformer_hyperparams["n_heads"]
    num_layers = transformer_hyperparams["num_layers"]
    dim_feedforward = transformer_hyperparams["dim_feedforward"]
    activation_name = transformer_hyperparams["activation_name"]
    pos_enc_type = transformer_hyperparams["pos_enc_type"]
    encoder_layer_type = transformer_hyperparams["encoder_layer_type"]
    dropout = transformer_hyperparams["dropout"]

    learning_rate = transformer_retraining["lr"]
    dropout = transformer_retraining["dropout"]
    weight_decay = transformer_retraining["weight_decay"]
    num_epochs = transformer_retraining["num_epochs"]
    batch_size = transformer_retraining["batch_size"]

    # Parameters
    _, source_len, feature_size = initial_train_source_original_all.shape
    _, label_size, _ = initial_train_target_original_first.shape

    ##############################
    # START RETRAINING THE MODEL #
    ##############################

    train_dataset = TensorDataset(combined_train_source_original_all, combined_train_target_original_first)
    test_dataset = TensorDataset(initial_test_source_original_all, initial_test_target_original_first)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
    # Initialize model, loss function, and optimizer
    
    model = TransformerEncoder(feature_size, label_size, source_len,
                    d_model, n_heads, num_layers, dim_feedforward, 
                    activation_name, pos_enc_type, encoder_layer_type,
                    dropout=dropout).to(device)
    
    criterion = RMSELoss() 
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # Adding L2 regularization
    
    # Count the number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print_log(f'The model has {total_params} parameters', log_path = log_path, file_name = training_log_file_name)
    
    best_test_loss = float('inf')
    if previous_iteration_index == 0:
        # Loading the best model from the previous training
        model.load_state_dict(torch.load(f"{models_path}/transformer/initial/{model_name}"))
    else:
        model.load_state_dict(torch.load(f"{models_path}/transformer/iteration_{previous_iteration_index}/{model_name}"))
    
    # Create output folder
    os.makedirs(f"{models_path}/transformer/iteration_{current_iteration_index}", exist_ok=True)
    # Track the best model
    best_model_path = f"{models_path}/transformer/iteration_{current_iteration_index}/{model_name}"

    # Lists to track train and test losses
    train_losses = []
    test_losses = []
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
            
        for batch_idx, (source_batch, target_batch) in enumerate(train_loader):
            source_batch, target_batch = source_batch.to(device), target_batch.to(device)
    
            optimizer.zero_grad()
    
            # Forward pass
            outputs = model(source_batch)
            loss = criterion(outputs, target_batch)
    
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
    
            train_loss += loss.item()
    
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
    
        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for source_batch, target_batch in test_loader:
                source_batch, target_batch = source_batch.to(device), target_batch.to(device)
    
                # Forward pass
                outputs = model(source_batch)
                loss = criterion(outputs, target_batch)
    
                test_loss += loss.item()
    
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        
        # Save the best model
        if test_loss < best_test_loss:
            print_log(f"New best test loss found: {test_loss}", log_path = log_path, file_name = training_log_file_name)
            best_test_loss = test_loss
            best_model = model
            torch.save(model.state_dict(), best_model_path)
    
        # Print progress
        if (epoch+1) % 100 == 0:
            print_log(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.9f}, Test Loss: {test_loss:.9f}', log_path = log_path, file_name = training_log_file_name)
    
    # Save the train and test loss lists as .npy files
    np.save(f'{models_path}/transformer/iteration_{current_iteration_index}/train_losses.npy', np.array(train_losses))
    np.save(f'{models_path}/transformer/iteration_{current_iteration_index}/test_losses.npy', np.array(test_losses))
    
    print_log('Finish retraining transformer model', log_path = log_path, file_name = training_log_file_name)
    print_log(f'Best model saved with test loss: {best_test_loss:.9f}', log_path = log_path, file_name = training_log_file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retraining LSTM model for the next iteration")
    
    parser.add_argument('--chosen_project_path', type=str, required=True, help='chosen project path')
    parser.add_argument('--current_iteration_index', type=str, required=True, help='current iteration index')
    parser.add_argument('--previous_iteration_index', type=str, required=True, help='previous iteration index')
    args = parser.parse_args()
    
    # Retraining the model
    retrain_model(args.chosen_project_path, int(args.current_iteration_index), int(args.previous_iteration_index))


