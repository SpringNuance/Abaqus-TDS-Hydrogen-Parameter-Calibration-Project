
import numpy as np
import os
import torch
import argparse
import json
import sys

from LSTM_helper import *
from LSTM import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

    initial_train_source_diff_all = torch.load(f"{training_data_path}/initial_train_source_diff_all.pt")
    initial_train_target_diff_last = torch.load(f"{training_data_path}/initial_train_target_diff_last.pt")
    initial_test_source_diff_all = torch.load(f"{training_data_path}/initial_test_source_diff_all.pt")
    initial_test_target_diff_last = torch.load(f"{training_data_path}/initial_test_target_diff_last.pt")
    
    # Convert them to float32

    initial_train_source_diff_all = initial_train_source_diff_all.float()
    initial_train_target_diff_last = initial_train_target_diff_last.float()
    initial_test_source_diff_all = initial_test_source_diff_all.float()
    initial_test_target_diff_last = initial_test_target_diff_last.float()

    print_log(f"\nShape of initial_train_source_diff_all: {initial_train_source_diff_all.shape}", log_path = log_path, file_name = training_log_file_name)
    print_log(f"Shape of initial_train_target_diff_last: {initial_train_target_diff_last.shape}", log_path = log_path, file_name = training_log_file_name)
    print_log(f"Shape of initial_test_source_diff_all: {initial_test_source_diff_all.shape}", log_path = log_path, file_name = training_log_file_name)
    print_log(f"Shape of initial_test_target_diff_last: {initial_test_target_diff_last.shape}", log_path = log_path, file_name = training_log_file_name)

    # Check if any of them has NaN or infinite values

    print_log(f"\nNumber of NaN values in initial_train_source_diff_all: {torch.isnan(initial_train_source_diff_all).sum()}", log_path = log_path, file_name = training_log_file_name)
    print_log(f"Number of NaN values in initial_train_target_diff_last: {torch.isnan(initial_train_target_diff_last).sum()}", log_path = log_path, file_name = training_log_file_name)
    print_log(f"Number of NaN values in initial_test_source_diff_all: {torch.isnan(initial_test_source_diff_all).sum()}", log_path = log_path, file_name = training_log_file_name)
    print_log(f"Number of NaN values in initial_test_target_diff_last: {torch.isnan(initial_test_target_diff_last).sum()}", log_path = log_path, file_name = training_log_file_name)

    print_log(f"\nNumber of infinite values in initial_train_source_diff_all: {torch.isinf(initial_train_source_diff_all).sum()}", log_path = log_path, file_name = training_log_file_name)
    print_log(f"Number of infinite values in initial_train_target_diff_last: {torch.isinf(initial_train_target_diff_last).sum()}", log_path = log_path, file_name = training_log_file_name)
    print_log(f"Number of infinite values in initial_test_source_diff_all: {torch.isinf(initial_test_source_diff_all).sum()}", log_path = log_path, file_name = training_log_file_name)
    print_log(f"Number of infinite values in initial_test_target_diff_last: {torch.isinf(initial_test_target_diff_last).sum()}", log_path = log_path, file_name = training_log_file_name)

    # Ensure that all target_sequence are positive
    print_log(f"\nNumber of values <= in initial_train_target_diff_last: {(initial_train_target_diff_last < 0).sum()}", log_path = log_path, file_name = training_log_file_name)
    print_log(f"Number of values <= in initial_test_target_diff_last: {(initial_test_target_diff_last < 0).sum()}", log_path = log_path, file_name = training_log_file_name)

    # Ensure that the scale of the source sequence is correct
    print_log("The first time step of the initial_train_source_diff_all tensor is:", log_path = log_path, file_name = training_log_file_name)
    print_log(initial_train_source_diff_all[0][0], log_path = log_path, file_name = training_log_file_name)

    # Ensure that the scale of the target sequence is correct
    print_log("The first time step of the initial_train_target_diff_last tensor is:", log_path = log_path, file_name = training_log_file_name)
    print_log(initial_train_target_diff_last[0][0], log_path = log_path, file_name = training_log_file_name)
    

    # Loading the iteration training data
    iteration_train_source_diff_all = torch.load(f"{training_data_path}/iteration_train_source_diff_all.pt")
    iteration_train_target_diff_last = torch.load(f"{training_data_path}/iteration_train_target_diff_last.pt")
    
    # Convert them to float32
    iteration_train_source_diff_all = iteration_train_source_diff_all.float()
    iteration_train_target_diff_last = iteration_train_target_diff_last.float()

    print_log(f"\nShape of iteration_train_source_diff_all: {iteration_train_source_diff_all.shape}", log_path = log_path, file_name = training_log_file_name)
    print_log(f"Shape of iteration_train_target_diff_last: {iteration_train_target_diff_last.shape}", log_path = log_path, file_name = training_log_file_name)

    print_log(f"Number of NaN values in iteration_train_source_diff_all: {torch.isnan(iteration_train_source_diff_all).sum()}", log_path = log_path, file_name = training_log_file_name)
    print_log(f"Number of NaN values in iteration_train_target_diff_last: {torch.isnan(iteration_train_target_diff_last).sum()}", log_path = log_path, file_name = training_log_file_name)

    print_log(f"Number of infinite values in iteration_train_source_diff_all: {torch.isinf(iteration_train_source_diff_all).sum()}", log_path = log_path, file_name = training_log_file_name)
    print_log(f"Number of infinite values in iteration_train_target_diff_last: {torch.isinf(iteration_train_target_diff_last).sum()}", log_path = log_path, file_name = training_log_file_name)

    print_log(f"Number of values <= 0 in iteration_train_target_diff_last: {(iteration_train_target_diff_last < 0).sum()}", log_path = log_path, file_name = training_log_file_name)
    
    print_log("The first time step of the iteration_train_source_diff_all tensor is:", log_path = log_path, file_name = training_log_file_name)
    print_log(iteration_train_source_diff_all[0][0], log_path = log_path, file_name = training_log_file_name)

    print_log("The first time step of the iteration_train_target_diff_last tensor is:", log_path = log_path, file_name = training_log_file_name)
    print_log(iteration_train_target_diff_last[0][0], log_path = log_path, file_name = training_log_file_name)

    # Now we would like to concatenate the initial and iteration training data
   
    combined_train_source_diff_all = torch.cat((initial_train_source_diff_all, iteration_train_source_diff_all), dim=0)
    combined_train_target_diff_last = torch.cat((initial_train_target_diff_last, iteration_train_target_diff_last), dim=0)

    print_log(f"\nShape of combined_train_source_diff_all: {combined_train_source_diff_all.shape}", log_path = log_path, file_name = training_log_file_name)
    print_log(f"Shape of combined_train_target_diff_last: {combined_train_target_diff_last.shape}", log_path = log_path, file_name = training_log_file_name)

    # Obtain the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_log(f"The training device: {device}", log_path = log_path, file_name = training_log_file_name)
    
    ###############################
    # Loading the hyperparameters #
    ###############################

    model_config = global_configs['model_config']
    
    LSTM_hyperparams = model_config['LSTM_hyperparams']
    LSTM_retraining = model_config['LSTM_retraining']
    model_name = LSTM_hyperparams["model_name"]

    hidden_size = LSTM_hyperparams["hidden_size"]
    num_layers = LSTM_hyperparams["num_layers"]
    bidirectional = LSTM_hyperparams["bidirectional"]
    use_attention = LSTM_hyperparams["use_attention"]
    attention_mechanism = LSTM_hyperparams["attention_mechanism"]

    start_lr = LSTM_retraining["start_lr"]
    end_lr = LSTM_retraining["end_lr"]
    dropout = LSTM_retraining["dropout"]
    lr_schedule = LSTM_retraining["lr_schedule"]
    start_tf = LSTM_retraining["start_tf"]
    end_tf = LSTM_retraining["end_tf"]
    tf_schedule = LSTM_retraining["tf_schedule"]
    weight_decay = LSTM_retraining["weight_decay"]
    num_epochs = LSTM_retraining["num_epochs"]
    batch_size = LSTM_retraining["batch_size"]
    max_ratio_differ = LSTM_retraining["max_ratio_differ"]

    _, source_len, feature_size = initial_train_source_diff_all.shape
    _, target_len, label_size = initial_train_target_diff_last.shape
    
    ##############################
    # START RETRAINING THE MODEL #
    ##############################
    
    train_dataset = TensorDataset(combined_train_source_diff_all, combined_train_target_diff_last)
    test_dataset = TensorDataset(initial_test_source_diff_all, initial_test_target_diff_last)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = LSTMModel(feature_size, label_size,
                    source_len, target_len,
                    hidden_size, num_layers,
                    dropout=dropout,
                    bidirectional=bidirectional, 
                    use_attention=use_attention).to(device)
    
    criterion = RMSELoss(linear_weight=True, max_ratio_differ = max_ratio_differ)  # Use the custom RMSE loss
    optimizer = optim.Adam(model.parameters(), lr=start_lr, weight_decay=weight_decay)  # Adding L2 regularization

    # Count the number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print_log(f'The model has {total_params} parameters', log_path = log_path, file_name = training_log_file_name)

    best_test_loss = float('inf')
    if previous_iteration_index == 0:
        # Loading the best model from the previous training
        model.load_state_dict(torch.load(f"{models_path}/LSTM/initial/{model_name}"))
    else:
        model.load_state_dict(torch.load(f"{models_path}/LSTM/iteration_{previous_iteration_index}/{model_name}"))
    
    # Create the output folder
    os.makedirs(f"{models_path}/LSTM/iteration_{current_iteration_index}", exist_ok=True)

    # Track the best model
    best_model_path = f"{models_path}/LSTM/iteration_{current_iteration_index}/{model_name}"
    
    # Lists to track train and test losses
    train_losses = []
    test_losses = []
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
    
        # Update learning rate
        if lr_schedule == 'linear':
            current_lr = linear_lr_scheduler(optimizer, epoch, start_lr, end_lr, num_epochs)
        else:
            current_lr = log_lr_scheduler(optimizer, epoch, start_lr, end_lr, num_epochs)
        
        # Update teacher forcing probability
        if tf_schedule == 'linear':
            teacher_forcing_prob = linear_teacher_forcing_scheduler(epoch, start_tf, end_tf, num_epochs)
        else:
            teacher_forcing_prob = log_teacher_forcing_scheduler(epoch, start_tf, end_tf, num_epochs)

        # teacher_forcing_prob = log_teacher_forcing_scheduler(epoch, start_tf, end_tf, num_epochs)
        for batch_idx, (source_batch, target_batch) in enumerate(train_loader):
            source_batch, target_batch = source_batch.to(device), target_batch.to(device)
    
            optimizer.zero_grad()
    
            # Forward pass
            outputs = model(source_batch, target_batch, teacher_forcing_prob)
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
            torch.save(model.state_dict(), best_model_path)

        # Print progress
        if (epoch+1) % 100 == 0:
            print_log(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.9f}, Test Loss: {test_loss:.9f}, LR: {current_lr:.9f}, TF: {teacher_forcing_prob:.9f}', log_path = log_path, file_name = training_log_file_name)
    
    # Save the train and test loss lists as .npy files
    np.save(f'{models_path}/LSTM/iteration_{current_iteration_index}/train_losses.npy', np.array(train_losses))
    np.save(f'{models_path}/LSTM/iteration_{current_iteration_index}/test_losses.npy', np.array(test_losses))
    
    print_log('Finish retraining LSTM model', log_path = log_path, file_name = training_log_file_name)
    print_log(f'Best model saved with test loss: {best_test_loss:.9f}', log_path = log_path, file_name = training_log_file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retraining LSTM model for the next iteration")
    
    parser.add_argument('--chosen_project_path', type=str, required=True, help='chosen project path')
    parser.add_argument('--current_iteration_index', type=str, required=True, help='current iteration index')
    parser.add_argument('--previous_iteration_index', type=str, required=True, help='previous iteration index')
    args = parser.parse_args()
    
    # Retraining the model
    retrain_model(args.chosen_project_path, int(args.current_iteration_index), int(args.previous_iteration_index))


