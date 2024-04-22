from torch import nn
import torch

class MaskedDecoder(nn.modules.Module):
    """
    The FCN used to predict the masked values consists of a single linear layer of size 
    dmodel/2 = 128 followed by a ReLU activation function. 
    An additional linear layer is used to project the output vector to a single value, 
    which corresponds to the predicted value of a masked point
    """
    def __init__(self, d_model, feat_dim):
        super(MaskedDecoder, self).__init__()
        self.linear1 = nn.Linear(d_model, d_model // 2)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_model // 2, feat_dim)
        self.bn1 = nn.BatchNorm1d(d_model // 2)
        self.bn2 = nn.BatchNorm1d(1)	
       # self.output_layer = nn.Linear(d_model, feat_dim)


    def forward(self, x):
         # from encoder [seq_len, batch_size, d_model] (We want to reduce the d_model to 1)
        #print(x.shape)
        #x = x.permute(1, 0, 2) # [seq_len, batch_size, d_model]
        #print(x.shape)
        x = self.linear1(x)
        #x = x.permute(1, 2, 0)  # [batch_size, d_model // 2, seq_len]
        #x = self.bn1(x)
        #x = x.permute(2, 0, 1)
        x = self.relu(x)
        x = self.linear2(x)
        #x = x.permute(1, 0, 2)
        #x = self.output_layer(x)
        return x # [seq_len, batch_size]

class ERP_decoder(nn.Module): 
    """
    This class defines the decoder of the ERP model. It takes the output of the encoder and returns the prediction.
    """
    def __init__(self, d_model, ts_steps, dropout=0.2):
        super(ERP_decoder, self).__init__()
        self.linear = nn.Linear(d_model, 1)
        self.linear2 = nn.Linear(ts_steps,1)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(ts_steps)
        self.bn2 = nn.BatchNorm1d(1)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # We describe the size of the tensors in each step
        # (batch_size, ts_steps, d_model)
        x = self.linear(x) # (batch_size, ts_steps, 1)
        x = self.bn1(x) # (batch_size, ts_steps, 1)
        x= x.squeeze(-1) # (batch_size, ts_steps)
        x = self.relu1(x) # (batch_size, ts_steps)
        x = self.dropout(x)
        x = self.linear2(x) # (batch_size, 1)
        x = self.bn2(x)
        x = x.squeeze(-1) # (batch_size)
        #x = self.sigmoid(x)
        return x
    
class Emotion_decoder(nn.Module): 
    """
    This class defines the decoder of the Emotion model. It takes the output of the encoder and returns the prediction.
    """
    def __init__(self, d_model,ts_steps, n_emotions, dropout=0.1):
        super(Emotion_decoder, self).__init__()
        self.linear = nn.Linear(d_model, 1)
        self.linear2 = nn.Linear(ts_steps,n_emotions)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(ts_steps)
        self.bn2 = nn.BatchNorm1d(n_emotions)
        self.dropout1 = nn.Dropout(dropout)
    
    def forward(self, x):
        # We describe the size of the tensors in each step
        # (batch_size, ts_steps, d_model)
        x = self.linear(x)
        x= x.squeeze(-1)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        return x
