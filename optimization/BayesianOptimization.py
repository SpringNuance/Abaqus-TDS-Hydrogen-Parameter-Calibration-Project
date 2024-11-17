import torch
from typing import Optional, Any
from torch import nn, Tensor
from torch.nn import functional as F
import numpy as np
import time
import os
import sys
import math

from botorch.sampling.normal import SobolQMCNormalSampler, IIDNormalSampler
from botorch.acquisition import qExpectedImprovement, qProbabilityOfImprovement, qUpperConfidenceBound
from botorch.optim import optimize_acqf

from utils.IO import print_log

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# use a GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

class BayesianOptimizationWrapper:
    def __init__(self, all_paths, GP_model, train_Y, 
                 bounds, param_config, optimization_config):
        
        # Initialize the Gaussian Process model
        self.GP_model = GP_model
        self.train_Y = train_Y
        self.bounds = bounds
        self.param_config = param_config
        self.optimization_config = optimization_config
        self.all_paths = all_paths
        self.log_path = self.all_paths['log_path']

        # Set best observed value for acquisition functions
        self.best_f = train_Y.max()

        # Set up the sampler based on config
        chosen_sampler = optimization_config['chosen_sampler']
        sample_shape = optimization_config['sample_shape']
        if chosen_sampler == "SobolQMCNormalSampler":
            self.sampler = SobolQMCNormalSampler(sample_shape=torch.Size([sample_shape]))
        elif chosen_sampler == "IIDNormalSampler":
            self.sampler = IIDNormalSampler(sample_shape=torch.Size([sample_shape]))
        else:
            raise ValueError("Unsupported sampler type. Choose either 'SobolQMCNormalSampler' or 'IIDNormalSampler'.")

        # Set up the acquisition function based on config
        
        chosen_acq = self.optimization_config["chosen_acq"]
        chosen_acq_config = self.optimization_config[chosen_acq]
        if chosen_acq == "qExpectedImprovement":
            eta = chosen_acq_config["eta"]
            self.acq_function = qExpectedImprovement(model=self.GP_model, best_f=self.best_f, 
                                                      sampler=self.sampler, eta=eta)

        elif chosen_acq == "qProbabilityOfImprovement":
            eta = chosen_acq_config["eta"]
            tau = chosen_acq_config["tau"]
            self.acq_function = qProbabilityOfImprovement(model=self.GP_model, best_f=self.best_f, 
                                                           sampler=self.sampler, tau=tau, eta=eta)

        elif chosen_acq == "qUpperConfidenceBound":
            beta = chosen_acq_config["beta"]
            self.acq_function = qUpperConfidenceBound(model=self.GP_model, 
                                            sampler=self.sampler, beta=beta)

        else:
            raise ValueError("Unsupported acquisition function. Choose one of 'qExpectedImprovement', 'qProbabilityOfImprovement', or 'qUpperConfidenceBound'.")

    def optimize_acq_function(self):
        
        """
            Optimize the acquisition function to suggest the next best point(s).
        """

        acq_function = self.acq_function
        return_best_only = self.optimization_config["return_best_only"]
        sequential = self.optimization_config["sequential"]
        q = self.optimization_config["q"]
        num_restarts = self.optimization_config["num_restarts"]
        raw_samples = self.optimization_config["raw_samples"]
        options = self.optimization_config["options"]
        
        # Optimize acquisition function to get the next candidate point(s)
        candidates, acq_values = optimize_acqf(
            acq_function=acq_function,
            bounds=self.bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            return_best_only=return_best_only,
            sequential=sequential,
            options=options,
        )

        candidates = candidates.cpu().numpy()
        acq_values = acq_values.cpu().numpy()

        return candidates, acq_values

    def get_acquisition_value(self, acq_function, suggested_points):
        """Evaluate acquisition function values at given points."""
        suggested_points = torch.tensor(suggested_points, dtype=dtype, device=device)
        acq_values = []
        with torch.no_grad():
            for point in suggested_points:
                point_reshaped = point.unsqueeze(0).unsqueeze(0)
                acq_values.append(acq_function(point_reshaped).item())
        return np.array(acq_values)

    def evaluate_acqf(self, grid_points):
        """Evaluate acquisition function on a grid of points."""
        with torch.no_grad():
            grid_points_reshaped = grid_points.unsqueeze(1)
            return self.acq_function(grid_points_reshaped).cpu().numpy()
