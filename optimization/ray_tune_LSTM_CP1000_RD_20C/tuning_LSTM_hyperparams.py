#!/usr/bin/env python
# coding: utf-8

##########################################
# RayTune Hyperparameter Tuning for LSTM #
##########################################

import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter

from filelock import FileLock
import tempfile
from typing import Dict

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import os
from math import *
from torch.utils.data import DataLoader, TensorDataset
import torch

if not os.getcwd().endswith("Abaqus-Hardening-Seq-2-Seq-Project"):
    # Move up two directories
    path_parent = os.path.dirname(os.getcwd())
    os.chdir(path_parent)
    path_parent = os.path.dirname(os.getcwd())
    os.chdir(path_parent)

from configs.chosen_project import *
from src.stage1_global_configs import *

chosen_project_path = "configs/global_config_CP1000_RD_20C.json"

global_configs = main_global_configs(chosen_project_path)

all_paths = global_configs['all_paths']
models_path = all_paths['models_path']
objectives = global_configs['objectives']
training_data_path = all_paths['training_data_path']

# -----------------------------------
# Data loaders
# We wrap the data loaders in their own function and pass a global data directory. 
# This way we can share a data directory between different trials.
# -----------------------------------

def load_training_testing_data(batch_size_train=32, batch_size_test=64, shuffle_train=True):
   
    # We add FileLock here because multiple workers will want to
    # load data, and this may cause overwrites since
    # DataLoader is not threadsafe.
    # with FileLock(os.path.expanduser("optimization/ray_tune_LSTM_CP1000_RD_20C/.data.lock")):

    initial_train_source_sequence_diff = torch.load(f"{training_data_path}/initial_train_source_sequence_diff.pt").float()
    initial_train_target_sequence_diff = torch.load(f"{training_data_path}/initial_train_target_sequence_diff.pt").float()
    initial_test_source_sequence_diff = torch.load(f"{training_data_path}/initial_test_source_sequence_diff.pt").float()
    initial_test_target_sequence_diff = torch.load(f"{training_data_path}/initial_test_target_sequence_diff.pt").float()

    # Convert them to float32

    initial_train_source_sequence_diff = initial_train_source_sequence_diff.float()
    initial_train_target_sequence_diff = initial_train_target_sequence_diff.float()
    initial_test_source_sequence_diff = initial_test_source_sequence_diff.float()
    initial_test_target_sequence_diff = initial_test_target_sequence_diff.float()

    # Create TensorDatasets
    train_dataset = TensorDataset(initial_train_source_sequence_diff, initial_train_target_sequence_diff)
    test_dataset = TensorDataset(initial_test_source_sequence_diff, initial_test_target_sequence_diff)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=shuffle_train)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

    return train_loader, test_loader

# -----------------------------------
# We can only tune those parameters that are configurable
# -----------------------------------

from optimization.LSTM_helper import *
from optimization.LSTM import *

def train_LSTM(searching_space: Dict):
    train_loader, test_loader = load_training_testing_data()

    # Parameters
    initial_train_source_sequence_diff = torch.load(f"{training_data_path}/initial_train_source_sequence_diff.pt").float()
    initial_train_target_sequence_diff = torch.load(f"{training_data_path}/initial_train_target_sequence_diff.pt").float()
    batch_size_train, source_len, feature_size = initial_train_source_sequence_diff.shape
    batch_size_test, target_len, label_size = initial_train_target_sequence_diff.shape

    hidden_size = searching_space["hidden_size"]
    num_layers = searching_space["num_layers"]
    dropout = searching_space["dropout"]
    weight_decay = searching_space["weight_decay"]
    attention_mechanism = searching_space["attention_mechanism"]

    start_lr = 0.001
    end_lr = 0.00001

    start_tf = 1.0
    end_tf = 0.1

    # Track the best model
    best_test_loss = float('inf')
    best_model_path = f"{models_path}/LSTM/initial_model.pth"
    best_model = None

    model = LSTMModel(feature_size, label_size,
                        source_len, target_len,
                        hidden_size, num_layers,
                        dropout=dropout,
                        bidirectional=True, 
                        use_attention=True,
                        attention_mechanism=attention_mechanism,
                        ).to(device)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    criterion = RMSELoss(linear_weight=True, max_ratio_differ = 5)  # Use the custom RMSE loss
    optimizer = optim.Adam(model.parameters(), 
                           lr=start_lr, 
                           weight_decay=weight_decay)  # Adding L2 regularization

    # Load existing checkpoint through `get_checkpoint()` API.
    if train.get_checkpoint():
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

    # Lists to track train and test losses
    
    train_losses = []
    test_losses = []

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):

        model.train()
        train_loss = 0.0
        test_loss = 0.0

        # Update learning rate
        current_lr = linear_lr_scheduler(optimizer, epoch, start_lr, end_lr, num_epochs)
        
        # Get the current teacher forcing probability from the scheduler
        teacher_forcing_prob = linear_teacher_forcing_scheduler(epoch, start_tf, end_tf, num_epochs)

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

        train_loss_epoch_average /= len(train_loader)

        # Evaluate on test set
        model.eval()
        
        with torch.no_grad():
            for source_batch, target_batch in test_loader:
                source_batch, target_batch = source_batch.to(device), target_batch.to(device)

                # Forward pass
                outputs = model(source_batch)
                loss = criterion(outputs, target_batch)

                test_loss += loss.item()

        test_loss_epoch_average /= len(test_loader)

        # Print progress
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, LR: {current_lr:.6f}, TF: {teacher_forcing_prob:.6f}')

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and will potentially be accessed through in ``get_checkpoint()``
        # in future iterations.
        # Note to save a file like checkpoint, you still need to put it under a directory
        # to construct a checkpoint.
        
        temp_checkpoint_dir = "optimization/ray_tune_LSTM_CP1000_RD_20C/checkpoint"
        path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
        torch.save(
            (model.state_dict(), optimizer.state_dict()), path
        )
        checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
        train.report(
            {"loss": test_loss_epoch_average},
            checkpoint=checkpoint,
        )

    print("Finished training")


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    search_space = {
        "num_layers": tune.sample_from(1, 2),  # Tune from 1-2 layers
        "hidden_size": tune.sample_from(50, 500),  # Tune from 50-500 hidden units
        "dropout": tune.loguniform(1e-6, 1e-1),
        "weight_decay": tune.loguniform(1e-6, 1e-1),
        "attention_mechanism": tune.choice(["dot", "general", "concat"]),
    }

    # Here we define the Optuna search algorithm:
    algo = OptunaSearch()

    # We also constrain the number of concurrent trials to 4 with a ConcurrencyLimiter
    algo = ConcurrencyLimiter(algo, max_concurrent=4)

    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_LSTM),
            resources={"cpu": 2, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=search_space,
    )
    results = tuner.fit()
    
    best_result = results.get_best_result(metric="loss", 
                                          mode="min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))

main(num_samples=2, max_num_epochs=2, gpus_per_trial=0)






