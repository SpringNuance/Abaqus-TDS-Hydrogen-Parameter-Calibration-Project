import torch
from typing import Optional, Any
from torch import nn, Tensor
from torch.nn import functional as F
import numpy as np
import time
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from optimization.LSTM_helper import *

# LSTM Encoder-Decoder Model with Bidirection and Attention mechanism

class LSTMModel(nn.Module):
    def __init__(self, feature_size, label_size,
                       source_len, target_len,
                       hidden_size, num_layers,
                       dropout=0.01,
                       bidirectional=False, 
                       use_attention=False,
                       attention_mechanism='dot',
                       ):
        # Possible attention mechanisms: dot, general, concat
        # We dont use multihead attention, it is computationally expensive
        # That is reserved for Transformer model
        
        super(LSTMModel, self).__init__()
        
        self.feature_size = feature_size
        self.label_size = label_size
        self.source_len = source_len
        self.target_len = target_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention

        # Encoder LSTM (can be unidirectional or bidirectional)
        self.encoder_lstm = nn.LSTM(input_size=feature_size, hidden_size=hidden_size, 
                                    num_layers=num_layers, batch_first=True, 
                                    dropout=dropout,
                                    bidirectional=bidirectional)

        # Decoder LSTM (must only be unidirectional)
        self.decoder_lstm = nn.LSTM(input_size=label_size, hidden_size=hidden_size * 2 if bidirectional else hidden_size, 
                                    num_layers=num_layers, batch_first=True, 
                                    dropout=dropout,
                                    bidirectional=False)
        
        # The final linear layer that maps decoded LSTM hidden size to the label size
        self.fc_final = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, label_size)
        
        # Activation function to enforce positive increments
        self.relu = nn.ReLU()
        
        if self.use_attention:
            self.attention = Attention(hidden_size * 2 if bidirectional else hidden_size,
                                       method=attention_mechanism)
            self.layer_norm = nn.LayerNorm(hidden_size * (2 if self.bidirectional else 1))

    def forward(self, source_seq, target_seq=None, teacher_forcing_prob=0.0):
        """
            Param: source_seq of shape [batch_size, source_len, feature_size]
            Return: target_seq of shape [batch_size, target_len, label_size]
        """
        # Initializing h_0 and c_0 for the encoder LSTM
        batch_size = source_seq.size(0)

        h_0_encoder = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), 
                        batch_size, self.hidden_size).to(source_seq.device)
        c_0_encoder = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), 
                        batch_size, self.hidden_size).to(source_seq.device)
        
        # if bidirectional and num_layers = 2, then h_0_encoder shape [4, batch_size, hidden_size]
        # h_0_encoder[0] is the forward hidden state of the first layer
        # h_0_encoder[1] is the backward hidden state of the first layer
        # h_0_encoder[2] is the forward hidden state of the second layer
        # h_0_encoder[3] is the backward hidden state of the second layer
        # The same for c_0_encoder

        encoder_outputs, (h_n_encoder, c_n_encoder) = self.encoder_lstm(source_seq, (h_0_encoder, c_0_encoder))
        
        # encoder_outputs shape [batch_size, source_len, hidden_size * (2 if self.bidirectional else 1)]
        # encoder_outputs[:, :, 0] corresponds to the forward hidden states
        # encoder_outputs[:, :, 1] corresponds to the backward hidden states
        
        #########################################
        # LSTM decoder with attention mechanism #
        #########################################

        # Initialize a list to hold the decoder outputs
        decoder_outputs = []
        
        if self.use_attention:

            # Prepare the initial input for the decoder LSTM (e.g., could be a vector of zeros)
            # 1 means the decoder will output one time step gradually
            decoder_input = torch.zeros(batch_size, 1, self.label_size).to(source_seq.device)  
            
            # In attention mechanism, we dont need to receive the contextual data of hidden and cell state
            # The attention mechanism would be able to get the contextual data from the encoder_outputs alone
            
            h_t_decoder = torch.zeros(self.num_layers, batch_size, 
                                      self.hidden_size * 2 if self.bidirectional else self.hidden_size
                                      ).to(source_seq.device)
            c_t_decoder = torch.zeros(self.num_layers, batch_size, 
                                      self.hidden_size * 2 if self.bidirectional else self.hidden_size
                                      ).to(source_seq.device)
                    
            # Iterate over the number of steps over target_len
            for t in range(self.target_len):
                
                # Pass the current input and the last hidden and cell states into the decoder LSTM
                decoder_output, (h_t_decoder, c_t_decoder) = self.decoder_lstm(decoder_input, (h_t_decoder, c_t_decoder))
                # decoder_output shape [batch_size, 1, hidden_size * (2 if self.bidirectional else 1)]
                # print(encoder_outputs.size(), decoder_output.size())
                # [32, 98, 10], [32, 1, 5]
                # Attention needs the current decoder output and all encoder outputs
                context_vector, attention_weights = self.attention(decoder_output, encoder_outputs)
               
                # context vector shape [batch_size, 1, hidden_size * (2 if self.bidirectional else 1)]
                # attention_weights shape [batch_size, 1, source_len]
                
                # Add the context vector directly to the decoder output
                decoder_output = decoder_output + context_vector
                # Apply layer normalization
                decoder_output = self.layer_norm(decoder_output)

                # Pass the output through the fully connected layer
                decoder_output = self.fc_final(decoder_output.squeeze(1))
                # decoder_output shape [batch_size, 1]
                
                # Apply ReLU activation (target sequence value is always positive)
                decoder_output = self.relu(decoder_output)
                # decoder_output shape [batch_size, 1]

                # Store the output for each time step
                decoder_outputs.append(decoder_output.unsqueeze(1))

                # Decide whether to use teacher forcing
                if target_seq is not None and torch.rand(1).item() < teacher_forcing_prob:
                    decoder_input = target_seq[:, t].unsqueeze(1)
                else:
                    decoder_input = decoder_output.unsqueeze(1)

            # Concatenate all decoder outputs along the time dimension
            
            target_seq = torch.cat(decoder_outputs, dim=1)

        ############################################
        # LSTM decoder without attention mechanism #
        ############################################

        else:

            # Prepare the initial input for the decoder LSTM (e.g., could be a vector of zeros)
            # 1 means the decoder will output one time step gradually
            decoder_input = torch.zeros(batch_size, 1, self.label_size).to(source_seq.device)  
                
            # Wihout attention mechanism, we need to receive the contextual data of last hidden and cell state
            # Because the Decoder cannot have access to encoder_outputs directly
            
            if self.bidirectional:
                # Reshape h_t_decoder, c_t_decoder from [num_layers * 2, batch_size, hidden_size]
                #                                    to [num_layers, batch_size, hidden_size * 2]
                h_t_decoder = h_n_encoder.view(self.num_layers, 2, batch_size, self.hidden_size)
                h_t_decoder = torch.cat((h_t_decoder[:, 0, :, :], h_t_decoder[:, 1, :, :]), dim=2)
                c_t_decoder = c_n_encoder.view(self.num_layers, 2, batch_size, self.hidden_size)
                c_t_decoder = torch.cat((c_t_decoder[:, 0, :, :], c_t_decoder[:, 1, :, :]), dim=2)
            else:
                h_t_decoder = h_n_encoder
                c_t_decoder = c_n_encoder
            
            # Iterate over the number of steps over target_len
            for t in range(self.target_len):
                
                # Pass the current input and the last hidden and cell states into the decoder LSTM
                decoder_output, (h_t_decoder, c_t_decoder) = self.decoder_lstm(decoder_input, (h_t_decoder, c_t_decoder))
                # decoder_output shape [batch_size, 1, hidden_size * (2 if self.bidirectional else 1)]
                
                # Pass the output through the fully connected layer
                decoder_output = self.fc_final(decoder_output.squeeze(1))
                # decoder_output shape [batch_size, 1]
                
                # Apply ReLU activation
                decoder_output = self.relu(decoder_output)
                # decoder_output shape [batch_size, 1]

                # Store the output for each time step
                decoder_outputs.append(decoder_output.unsqueeze(1))

                # Decide whether to use teacher forcing
                if target_seq is not None and torch.rand(1).item() < teacher_forcing_prob:
                    decoder_input = target_seq[:, t].unsqueeze(1)
                else:
                    decoder_input = decoder_output.unsqueeze(1)
                    
                # Use the output as the next input to the decoder
                decoder_input = decoder_output.unsqueeze(1)

            # Concatenate all decoder outputs along the time dimension
            
            target_seq = torch.cat(decoder_outputs, dim=1)
        # time.sleep(180)
        return target_seq
    
    