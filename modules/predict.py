import numpy as np
import pandas as pd
import torch


def seq2seq_predict_without_referenced_flow_curve(transformer_model, LSTM_model, 
                                                  exp_source_original_all, 
                                                  exp_source_diff_all, 
                                                  scale_source, scale_target):
    """
    Predict the true stress of the flow curve using Seq2Seq models
    Inference is done on CPU, assuming the models are already trained and loaded
    
    Parameters:
    
    transformer_model (Pytorch model): receives source_original_all sequences of shape [batch_size, source_len_original_all, num_objectives]
                       where source_len_original_all has same length as exp_source_original_all
                       and outputs target_original_first sequences of shape [batch_size, divided_index+1, num_objectives]
    
    LSTM_model (Pytorch model): receives source_diff_all sequences of shape [batch_size, source_len_diff_all, num_objectives]
                        where source_len_diff_all has same length as exp_source_diff_all
                        and outputs target_diff_last sequences of shape [batch_size, source_len_diff_all - divided_index, num_objectives]
                        Note is that source_len_diff_all - divided_index = source_len_original_all - divided_index - 1
    
    exp_source_original_all (torch.tensor): the unscaled original source sequence of shape [batch_size, source_len_original_all, num_objectives]:
    exp_source_diff_all (torch.tensor): the unscaled differenced source sequence of shape [batch_size, source_len_diff_all, num_objectives]:
    
    scale_source (float): the scale of the source sequence
    scale_target (float): the scale of the target sequence
    """
    
    # First, we need to assert the correct shape
    # batch size of must be similar
    assert exp_source_original_all.shape[0] == exp_source_diff_all.shape[0], "The batch size of the source original and differenced sequences must be the same"
    assert exp_source_original_all.shape[1] == exp_source_diff_all.shape[1] + 1, "The length of the source original sequence must be one more than the length of the source differenced sequence"   
    assert exp_source_original_all.shape[2] == exp_source_diff_all.shape[2], "The number of objectives of the source original and differenced sequences must be the same"

    transformer_model.to("cpu")
    LSTM_model.to("cpu")
    
    scaled_exp_source_original_all = exp_source_original_all * scale_source
    scaled_exp_source_diff_all = exp_source_diff_all * scale_source

    # These tensors can be in float64 or float32, so we need to ensure they are all in float32

    scaled_exp_source_original_all = scaled_exp_source_original_all.to(torch.float32)
    scaled_exp_source_diff_all = scaled_exp_source_diff_all.to(torch.float32)
    
    scaled_exp_source_original_all.to("cpu")
    scaled_exp_source_diff_all.to("cpu")
    
    # Ensure the models are in evaluation mode
    transformer_model.eval()
    LSTM_model.eval()

    with torch.no_grad(): 
        scaled_exp_target_original_first = transformer_model(scaled_exp_source_original_all)
        scaled_exp_target_diff_last = LSTM_model(scaled_exp_source_diff_all)
  
    scaled_exp_target_original_last = torch.zeros(scaled_exp_target_diff_last.shape)
    scaled_exp_target_original_last[:, 0, :] = scaled_exp_target_original_first[:, -1, :]
    
    target_len_diff_all = scaled_exp_target_diff_last.shape[1]
    for i in range(1, target_len_diff_all):
        scaled_exp_target_original_last[:, i, :] = scaled_exp_target_original_last[:, i-1, :] + scaled_exp_target_diff_last[:, i, :]
    
    # Combine the predictions
    # The last output of transformer is at divided_index + 1, which overlaps with the first output of LSTM
    scaled_exp_target_original_all = torch.cat((scaled_exp_target_original_first, scaled_exp_target_original_last), dim=1)
    
    # Scale the predictions back to the original scale
    exp_target_original_all = scaled_exp_target_original_all / scale_target
    return exp_target_original_all

def seq2seq_predict_with_referenced_flow_curve(referenced_exp_target_original_first, LSTM_model, 
                                               exp_source_diff_all, scale_source, scale_target):
    """
    Predict the true stress of the flow curve using Seq2Seq models
    Inference is done on CPU, assuming the models are already trained and loaded
    
    Parameters:
    
    referenced_exp_target_original_first (torch.tensor): the referenced flow curve stress values for the first stress values
                                                shape [batch_size, target_len_original_first, num_objectives]
    We assume that referenced_exp_target_original_first * scale_reference has the same scale as the exp_target_diff_last
                                                
    LSTM_model (Pytorch model): receives source_diff_all sequences of shape [batch_size, source_len_diff_all, num_objectives]
                        where source_len_diff_all has same length as exp_source_diff_all
                        and outputs target_diff_last sequences of shape [batch_size, source_len_diff_all - divided_index, num_objectives]
                        Note is that source_len_diff_all - divided_index = source_len_original_all - divided_index - 1

    exp_source_diff_all (torch.tensor): the unscaled differenced source sequence of shape [1, source_len_diff_all, num_objectives]
    
    scale_source (float): the scale of the source sequence
    scale_target (float): the scale of the target sequence
    """

    LSTM_model.to("cpu")

    scaled_referenced_exp_target_original_first = referenced_exp_target_original_first * scale_target

    scaled_exp_source_diff_all = exp_source_diff_all * scale_source

    # These tensors can be in float64 or float32, so we need to ensure they are all in float32

    scaled_referenced_exp_target_original_first = scaled_referenced_exp_target_original_first.to(torch.float32)
    scaled_exp_source_diff_all = scaled_exp_source_diff_all.to(torch.float32)

    scaled_referenced_exp_target_original_first.to("cpu")
    scaled_exp_source_diff_all.to("cpu")
    
    # Ensure the models are in evaluation mode
    LSTM_model.eval()

    with torch.no_grad(): 
        scaled_exp_target_diff_last = LSTM_model(scaled_exp_source_diff_all)
    
    scaled_exp_target_original_last = torch.zeros(scaled_exp_target_diff_last.shape)
    scaled_exp_target_original_last[:, 0, :] = scaled_referenced_exp_target_original_first[:, -1, :]
    
    target_len_diff_all = scaled_exp_target_diff_last.shape[1]
    for i in range(1, target_len_diff_all):
        scaled_exp_target_original_last[:, i, :] = scaled_exp_target_original_last[:, i-1, :] + scaled_exp_target_diff_last[:, i, :]
    
    # Combine the predictions
    # The last output of transformer is at divided_index + 1, which overlaps with the first output of LSTM
    
    scaled_exp_target_original_all = torch.cat((scaled_referenced_exp_target_original_first, scaled_exp_target_original_last), dim=1)
    
    # Scale the predictions back to the original scale
    exp_target_original_all = scaled_exp_target_original_all / scale_target

    return exp_target_original_all

def seq2seq_predict(transformer_model, LSTM_model, use_referenced_flow_curve, 
                    referenced_exp_target_original_first, 
                    exp_source_original_all, exp_source_diff_all,
                    scale_source, scale_target):
    """
    Predict the true stress of the flow curve using Seq2Seq models
    How it works: 
    - If use_referenced_flow_curve is True or transformer_model is None, then the Transformer model will not be used to predict the first <divided index> stress values
    Instead, we would directly use the referenced_flow_curve_stress values for the first <divided index> stress values
    - If use_referenced_flow_curve is False, then the Transformer model will be used to predict the first <divided index> stress values
    - If both use_referenced_flow_curve is True and transformer_model is not None, then we would use referenced_exp_target_original_first for the first <divided index> stress values

    The function would throw an error for these options:
    - use_referenced_flow_curve is False and transformer_model is None
    - use_referenced_flow_curve is True and referenced_exp_target_original_first is None
    """
    if use_referenced_flow_curve is False:
        assert transformer_model is not None, "The transformer_model must be provided if use_referenced_flow_curve is False"
        exp_target_original_all =\
            seq2seq_predict_without_referenced_flow_curve(transformer_model, LSTM_model, 
                                                          exp_source_original_all, 
                                                          exp_source_diff_all, 
                                                          scale_source, scale_target)
    
    if use_referenced_flow_curve is True:
        assert referenced_exp_target_original_first is not None, "The referenced_exp_target_original_first must be provided if use_referenced_flow_curve is True"
        exp_target_original_all =\
            seq2seq_predict_with_referenced_flow_curve(referenced_exp_target_original_first, 
                                                       LSTM_model, exp_source_diff_all, 
                                                       scale_source, scale_target)

    return exp_target_original_all

        