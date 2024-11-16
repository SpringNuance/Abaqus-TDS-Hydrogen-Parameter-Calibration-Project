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
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.optim import Adam

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# use a GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

class GaussianProcess():
    def __init__(self, chosen_kernel, kernel_params):

        self.chosen_kernel = chosen_kernel
        self.kernel_params = kernel_params
    
    def init_kernel():
        
    def init_model():

        

# Verify the shapes
print("train_X shape:", train_X.shape)  # Should be (number of samples, number of features)
print("train_Y shape:", train_Y.shape)  # Should be (number of samples, 1)



# Define the kernel explicitly as an RBF kernel wrapped in a ScaleKernel
# If we dont set the ard_num_dims, the kernel will assume that all input dimensions have the same lengthscale
# It is known as an isotropic kernel

GP_kernel = ScaleKernel(RBFKernel(ard_num_dims=train_X.shape[1])) + ConstantKernel()
GP_kernel.to(train_X)

GP_model = SingleTaskGP(train_X=train_X, train_Y=train_Y, 
                    train_Yvar = None,
                    likelihood = None,
                    covar_module=GP_kernel)
GP_model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))



mll = ExactMarginalLogLikelihood(likelihood=GP_model.likelihood, model=GP_model)
# set mll and all submodules to the specified dtype and device
mll = mll.to(train_X)



NUM_EPOCHS = 40000

max_learning_rate = 0.0005
min_learning_rate = 0.0001



GP_model.train()

def train(self, train_X, train_Y, NUM_EPOCHS, max_learning_rate, min_learning_rate, lr_decrement):
    # Hyperparameters in the Multidimensional RBF Kernel
    # In the multidimensional case, the RBF kernel has:
    # One output scale parameter sigma for the entire kernel, which controls the overall variance of the function values.
    # One lengthscale parameter per dimension controls the smoothness in the i-th input dimension.

    # Initialize optimizer with the maximum learning rate
    optimizer = Adam([{'params': GP_model.parameters()}], lr=max_learning_rate)

    # Calculate the decrement step for the learning rate
    lr_decrement = (max_learning_rate - min_learning_rate) / NUM_EPOCHS

    for epoch in range(NUM_EPOCHS):
        # clear gradients
        optimizer.zero_grad()
        # forward pass through the model to obtain the output MultivariateNormal
        output = GP_model(train_X)
        # Compute negative marginal log likelihood
        loss = - mll(output, GP_model.train_targets)
        # back prop gradients
        loss.backward()
        # print every 2000 iterations
        if (epoch + 1) % 2000 == 0:
            # Retrieve lengthscale, noise, and output scale for tracking
            scale_kernel = list(GP_model.covar_module.sub_kernels())[0]
            GP_kernel = list(GP_model.covar_module.sub_kernels())[1]
            # GP_kernel is also scale_kernel.base_kernel
            constant_kernel = list(GP_model.covar_module.sub_kernels())[2]
            lengthscales = GP_kernel.lengthscale.tolist()[0]
            noise = GP_model.likelihood.noise.tolist()
            sigma = math.sqrt(scale_kernel.outputscale.item())  # Compute σ from σ^2
            constant_value = constant_kernel.constant.item()     # Get the optimized constant value
            
            # Print kernel parameters
            print(
                f"Epoch {epoch+1:>3}/{NUM_EPOCHS} - Loss: {loss.item():>4.3f} ",
                f"lengthscale: {[round(ls, 6) for ls in lengthscales]} ",
                f"noise: {[round(n, 6) for n in noise]} ",
                f"sigma: {round(sigma, 6)} ",
                f"constant: {round(constant_value, 6)}"
            )

        # Update optimizer learning rate linearly
        for param_group in optimizer.param_groups:
            param_group['lr'] = max(min_learning_rate, max_learning_rate - lr_decrement * epoch)

        optimizer.step()