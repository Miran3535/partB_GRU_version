# gru_model.py
import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        
        super(GRUModel, self).__init__()
        # This defines the GRU layer (batch_first so batch dimension comes first in input)
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, 
                          num_layers=num_layers, batch_first=True)
        # Linear layer to map from GRU hidden state to output prediction
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
       
        # GRU forward pass. We only need the output sequence (ignore hidden state).
        # output shape: (batch, seq_length, hidden_dim)
        output_seq, _ = self.gru(x)  
        # Take the output of the final time step in the sequence
        final_step_output = output_seq[:, -1, :]   # shape: (batch, hidden_dim)
        # Apply linear layer to get prediction
        out = self.fc(final_step_output)           # shape: (batch, output_dim)
        return out
