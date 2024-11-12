
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np


# Custom RMSE Loss Function
class RMSELoss(nn.Module):
    def __init__(self, linear_weight=False, log_weight=False, max_ratio_differ=10):
        super(RMSELoss, self).__init__()
        if linear_weight and log_weight:
            raise ValueError("Both linear_weight and log_weight cannot be True simultaneously.")
        if linear_weight or log_weight:
            if max_ratio_differ < 1.0:
                raise ValueError("max_ratio_differ must be at least 1.0.")
        
        self.mse = nn.MSELoss(reduction='none')
        self.linear_weight = linear_weight
        self.log_weight = log_weight
        self.max_ratio_differ = max_ratio_differ

    def forward(self, y_pred, y_true):
        mse = self.mse(y_pred, y_true)
        target_len = y_pred.size(1)  # y_pred and y_true are of shape [batch_size, target_len, label_dim]
        
        if self.linear_weight or self.log_weight:
                        
            # Create weights
            if self.linear_weight:
                weights = torch.linspace(self.max_ratio_differ, 1, steps = target_len, device = y_pred.device)
            elif self.log_weight:
                weights = torch.logspace(self.max_ratio_differ, 1, steps = target_len, base=10, device = y_pred.device)
            
            # Normalize weights so they sum to 1 and the ratio of first to last matches max_ratio_differ
            weights = weights / weights.sum() * target_len
            
            # Apply weights
            weighted_mse = (mse * weights.unsqueeze(0).unsqueeze(-1)).mean()  # Adjust dimensions for broadcasting
        else:
            weighted_mse = mse.mean()
        
        return torch.sqrt(weighted_mse)


class Attention(nn.Module):
    def __init__(self, hidden_size, method='dot'):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.method = method

        # Depending on the attention scoring method, define required layers
        if method == 'general':
            self.attn = nn.Linear(hidden_size, hidden_size)
        elif method == 'concat':
            self.attn = nn.Linear(hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))
    
    def forward(self, decoder_hidden, encoder_outputs):
        """
        Forward pass of the attention layer.
        
        Args:
            decoder_hidden: the current decoder state, shape [batch_size, 1, hidden_size]
            encoder_outputs: all encoder outputs, shape [batch_size, source_len, hidden_size]
        
        Returns:
            context_vector: weighted sum of encoder_outputs, shape [batch_size, 1, hidden_size]
            attention_weights: attention weights, shape [batch_size, 1, source_len]
        """
        if self.method == 'dot':
            # Dot product attention
            attention_scores = torch.bmm(encoder_outputs, decoder_hidden.transpose(1, 2))
        elif self.method == 'general':
            # General attention: score = encoder_out * W * decoder_hidden
            energy = self.attn(encoder_outputs)  # shape [batch_size, source_len, hidden_size]
            attention_scores = torch.bmm(energy, decoder_hidden.transpose(1, 2))
        elif self.method == 'concat':
            # Concat attention: score = v * tanh(W * [encoder_out; decoder_hidden])
            length = encoder_outputs.size(1)
            decoder_hidden_expanded = decoder_hidden.expand(-1, length, -1)
            combined = torch.cat((encoder_outputs, decoder_hidden_expanded), dim=2)
            energy = torch.tanh(self.attn(combined))
            attention_scores = torch.bmm(energy, self.v.unsqueeze(0).unsqueeze(0).expand(encoder_outputs.size(0), -1, -1).transpose(1, 2))

        # Turn scores to weights
        # Shape: [batch_size, 1, source_len]
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Multiply the weights by the encoder outputs to get the context vector
        # [batch_size, 1, hidden_size] = bmm([batch_size, 1, source_len], [batch_size, source_len, hidden_size])
        context_vector = torch.bmm(attention_weights.transpose(1, 2), encoder_outputs)
        
        return context_vector, attention_weights
    
# Linear Learning Rate Scheduler
def linear_lr_scheduler(optimizer, epoch, start_lr, end_lr, total_epochs):
    lr = start_lr + (end_lr - start_lr) * (epoch / total_epochs)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

# Log Learning Rate Scheduler
def log_lr_scheduler(optimizer, epoch, start_lr, end_lr, total_epochs):
    if epoch == 0:
        lr = start_lr
    else:
        lr = start_lr * np.exp(np.log(end_lr / start_lr) * epoch / total_epochs)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def linear_teacher_forcing_scheduler(epoch, start_tf, end_tf, total_epochs):
    # Calculate the current teacher forcing probability based on the linear schedule
    tf_prob = start_tf + (end_tf - start_tf) * (epoch / total_epochs)
    return tf_prob

def log_teacher_forcing_scheduler(epoch, start_tf, end_tf, total_epochs):
    # Ensure end_tf is never zero to avoid log(0)
    if end_tf <= 0:
        end_tf = 1e-10  # Small positive number close to zero

    if epoch == 0:
        # Avoid log(0) issue by returning the starting teacher forcing probability at the first epoch
        tf_prob = start_tf
    else:
        # Calculate the current teacher forcing probability based on the exponential decay
        tf_prob = start_tf * np.exp(np.log(end_tf / start_tf) * epoch / total_epochs)
    return tf_prob