import torch
from typing import Optional, Any
from torch import nn, Tensor
from torch.nn import functional as F
import numpy as np
import time
import os
import sys
import math

from botorch.models import SingleTaskGP
from gpytorch.constraints import GreaterThan
from botorch.models.transforms import Normalize
from gpytorch.kernels import RBFKernel, ScaleKernel, ConstantKernel, MaternKernel
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood, LeaveOneOutPseudoLikelihood
from torch.optim import Adam, SGD
from utils.IO import print_log

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# use a GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

class GaussianProcessWrapper():
    def __init__(self, model_config, all_paths, train_X, train_Y):

        self.chosen_kernel = model_config['chosen_kernel']
        self.kernel_config = model_config[self.chosen_kernel]
        self.chosen_mean = model_config['chosen_mean']
        self.chosen_likelihood = model_config['chosen_likelihood']
        self.fixed_noise = model_config['fixed_noise']
        self.fixed_noise_value = model_config['fixed_noise_value']
        self.trainable_min_noise_constraint = model_config['trainable_min_noise_constraint']
        self.GP_training = model_config['GP_training']
        self.train_X = train_X
        self.train_Y = train_Y
        self.ard_num_dims = train_X.shape[1]
        self.all_paths = all_paths

        # Convert the training data to tensors if they are not already
        if not torch.is_tensor(train_X):
            self.train_X = torch.tensor(train_X, dtype=dtype, device=device)
        if not torch.is_tensor(train_Y):
            self.train_Y = torch.tensor(train_Y, dtype=dtype, device=device)
            if self.train_Y.dim() == 1:
                self.train_Y = self.train_Y.unsqueeze(-1)
        
        # print("Training data shape: ", self.train_X.shape)
        # print("Training labels shape: ", self.train_Y.shape)
        # print( self.train_X)
        # print( self.train_Y)
        # time.sleep(180)
    
        if self.chosen_kernel == 'RBFKernel':
            self.GP_kernel = RBFKernel(ard_num_dims=self.ard_num_dims)
        elif self.chosen_kernel == 'MaternKernel':
            self.GP_kernel = MaternKernel(nu=kernel_config["nu"], 
                                          ard_num_dims=self.ard_num_dims)
        else:
            raise ValueError("The chosen kernel is not supported. Please choose either 'RBFKernel' or 'MaternKernel'")
        if self.kernel_config['scale_cov_kernel'] == True:
            self.GP_kernel = ScaleKernel(self.GP_kernel)
        if self.kernel_config['constant_cov_kernel'] == True:
            self.GP_kernel = self.GP_kernel + ConstantKernel()
        self.GP_kernel.to(self.train_X)

        if self.fixed_noise == True:
            self.train_Yvar = torch.full_like(self.train_Y, self.fixed_noise_value)
        else:
            self.train_Yvar = None
        
        self.GP_model = SingleTaskGP(train_X=self.train_X, train_Y=self.train_Y, 
                    train_Yvar = self.train_Yvar,
                    covar_module=self.GP_kernel)

        if self.fixed_noise == False:
            self.GP_model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(self.trainable_min_noise_constraint))
        
        if self.chosen_mean == 'ConstantMean':
            self.GP_model.mean_module = ConstantMean()
        elif self.chosen_mean == 'ZeroMean':
            self.GP_model.mean_module = ZeroMean()
        elif self.chosen_mean == 'LinearMean':
            self.GP_model.mean_module = LinearMean()
        else:
            raise ValueError("The chosen mean is not supported. Please choose either 'ConstantMean', 'ZeroMean' or 'LinearMean'")

        if self.chosen_likelihood == 'ExactMarginalLogLikelihood':
            self.likelihood = ExactMarginalLogLikelihood(likelihood=self.GP_model.likelihood, model=self.GP_model)
        elif self.chosen_likelihood == 'LeaveOneOutPseudoLikelihood':
            self.likelihood = LeaveOneOutPseudoLikelihood(self.GP_model, self.train_X, self.train_Y)
        else:
            raise ValueError("The chosen likelihood is not supported. Please choose either 'ExactMarginalLogLikelihood' or 'LeaveOneOutPseudoLikelihood'")
        self.likelihood.to(self.train_X)
    
    def train_model(self):

        num_epochs = self.GP_training['num_epochs']
        start_learning_rate = self.GP_training['start_learning_rate']
        end_learning_rate = self.GP_training['end_learning_rate']
        weight_decay = self.GP_training['weight_decay']

        # raise error if start learning rate is less than end learning rate
        if start_learning_rate < end_learning_rate:
            raise ValueError("The start learning rate must be greater than or equal the end learning rate")

        if self.GP_training['chosen_optimizer'] == 'Adam':
            optimizer = Adam([{'params': self.GP_model.parameters()}], 
                        lr=start_learning_rate, weight_decay=weight_decay)
        elif self.GP_training['chosen_optimizer'] == 'SGD':
            optimizer = SGD([{'params': self.GP_model.parameters()}], 
                        lr=start_learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError("The chosen optimizer is not supported. Please choose either 'Adam' or 'SGD'")
        
        # Calculate the decrement step for the learning rate
        lr_decrement = (start_learning_rate - end_learning_rate) / num_epochs

        self.GP_model.train()

        # Hyperparameters in the Multidimensional RBF Kernel
        # In the multidimensional case, the RBF kernel has:
        # One output scale parameter sigma for the entire kernel, which controls the overall variance of the function values.
        # One lengthscale parameter per dimension controls the smoothness in the i-th input dimension.

        # NOTE: Unlike other neural networks, GP cannot be trained in batches
        # This is because an exact GP model defines a multivariate Gaussian distribution over 
        # the entire set of training outputs, which is characterized by a mean vector and a covariance matrix.

        training_log = self.GP_training['training_log']
        logging_every_epoch = self.GP_training['logging_every_epoch']

        log_path = self.all_paths['log_path']

        for epoch in range(num_epochs):
            # clear gradients
            optimizer.zero_grad()
            # forward pass through the model to obtain the output MultivariateNormal
            output = self.GP_model(self.train_X)
            # Compute negative marginal log likelihood
            loss = - self.likelihood(output, self.GP_model.train_targets)
            # back prop gradients
            loss.backward()
            # logging the training progress

            if training_log == True:
                if (epoch + 1) % logging_every_epoch == 0:
                    message = f"Epoch {epoch+1:>3}/{num_epochs} - Loss: {loss.item():>4.3f}"
                    
                    # Retrieve lengthscale, noise, and output scale for tracking
                    if self.kernel_config['scale_cov_kernel'] == True:
                        scale_kernel = list(self.GP_model.covar_module.sub_kernels())[0]
                        sigma = math.sqrt(scale_kernel.outputscale.item())
                        message += f" - sigma: {round(sigma, 6)}"
                        
                        if self.chosen_kernel == 'RBFKernel' or self.chosen_kernel == 'MaternKernel':
                            base_kernel = list(self.GP_model.covar_module.sub_kernels())[1]
                            lengthscales = base_kernel.lengthscale.tolist()[0]
                            message += f" - lengthscales: {[round(ls, 6) for ls in lengthscales]}"
                        
                        if self.kernel_config['constant_cov_kernel'] == True:
                            constant_kernel = list(self.GP_model.covar_module.sub_kernels())[2]
                            cov_constant_value = constant_kernel.constant.item()
                            message += f" - cov constant: {round(cov_constant_value, 6)}"
                    else:
                        if self.chosen_kernel == 'RBFKernel' or self.chosen_kernel == 'MaternKernel':
                            base_kernel = list(self.GP_model.covar_module.sub_kernels())[0]
                            lengthscales = base_kernel.lengthscale.tolist()[0]
                            message += f" - lengthscales: {[round(ls, 6) for ls in lengthscales]}"

                        if self.kernel_config['constant_cov_kernel'] == True:
                            constant_kernel = list(self.GP_model.covar_module.sub_kernels())[1]
                            cov_constant_value = constant_kernel.constant.item()
                            message += f" - cov constant: {round(cov_constant_value, 6)}"
                        
                    mean_constant_value = self.GP_model.mean_module.constant.item()
                    message += f" - mean constant: {round(mean_constant_value, 6)}"

                    noise = self.GP_model.likelihood.noise.tolist()[0]
                    message += f" - noise: {round(noise, 6)}"

                    print_log(message, log_path)

            # Update optimizer learning rate linearly
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(end_learning_rate, start_learning_rate - lr_decrement * epoch)

            optimizer.step()

    def predict(self, test_X):
        """Predict the output of the GP model for the test data"""
        if not torch.is_tensor(test_X):
            test_X = torch.tensor(test_X, dtype=dtype, device=device)
        self.GP_model.eval()
        with torch.no_grad():
            # Make predictions
            # Obtain the posterior distribution
            posterior = GP_RBF_model.posterior(test_X)
            mean = posterior.mean  # Predicted mean
            stddev = posterior.variance.sqrt()  # Predicted standard deviation (uncertainty)
            mean_no_grad = mean.cpu().numpy()
            stddev_no_grad = stddev.cpu().numpy()
        return mean, stddev
    
    def sample(self, num_samples, grid_points):
        """
            Sample from the GP model
        """
        # Set the model to evaluation mode
        self.GP_model.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Get the posterior distribution over the grid points
            posterior = self.GP_model(grid_points)
            
            # Sample num_samples draws from the posterior distribution
            samples = posterior.rsample(torch.Size([num_samples]))
            
        # Reshape samples to be (num_samples, num_points)
        return samples

    def print_state_dict(self):
        """Print the state dictionary of the GP model"""
        print(self.GP_model.state_dict())

    def save_model(self, saving_model_path):
        """Save the GP model"""
        torch.save(self.GP_model.state_dict(), saving_model_path)

    def load_model(self, loading_model_path):
        """Load the GP model"""
        self.GP_model.load_state_dict(torch.load(loading_model_path))
