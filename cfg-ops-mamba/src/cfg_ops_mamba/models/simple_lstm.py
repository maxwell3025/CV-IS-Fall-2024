import torch
from torch import nn

class SimpleLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        drop_prob=0.3,
    ) -> None:
        super(SimpleLSTM,self).__init__()
 
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
 
        #lstm
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        
        # dropout layer
        self.dropout = nn.Dropout(drop_prob)
    
        # linear layer
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)

        # sigmoid layer
        self.sig = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        length = x.shape[1]
        assert x.shape == (batch_size, length, self.input_dim)

        hidden = self.init_hidden(batch_size)
        out = self.fc1(x)
        assert out.shape == (batch_size, length, self.hidden_dim)

        out, hidden = self.lstm(out, hidden)
        assert out.shape == (batch_size, length, self.hidden_dim)
        
        # dropout and fully connected layer
        out = self.dropout(out)
        out = self.fc2(out)
        assert out.shape == (batch_size, length, self.output_dim)

        return out
        
    def init_hidden(self, batch_size):
        """ Initializes hidden state """
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        device = next(self.parameters()).device
        h0 = torch.zeros((1, batch_size, self.hidden_dim)).to(device)
        c0 = torch.zeros((1, batch_size, self.hidden_dim)).to(device)
        hidden = (h0,c0)
        return hidden
