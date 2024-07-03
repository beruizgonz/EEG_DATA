import math

from torch import nn
from torch.nn import functional as F
from torch.nn.modules import TransformerEncoderLayer

from .pos_encoding import get_pos_encoder
from .transformer import TransformerEncoderLayer, TransformerEncoder, TransformerBatchNormEncoderLayer

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))


class ConvNet(nn.Module):
    def __init__(self, input_time=1000, fs=128, ncha=8, filters_per_branch=8, scales_time=250, activation='ELU', dropout_rate=0.25):
        super(ConvNet, self).__init__()
        input_samples = int(input_time * fs / 1000)
        scale_samples = int(scales_time * fs / 1000)

        # First convolutional layer
        layer1 = [
             nn.Conv2d(1, filters_per_branch, kernel_size=(scale_samples, 1), padding='same'),
                nn.BatchNorm2d(filters_per_branch),
                getattr(nn, activation)(),
                nn.Dropout(dropout_rate),
        ]

        # Second convolutional layer
        layer2 = [
             nn.Conv2d(filters_per_branch, filters_per_branch*2, kernel_size=(1, ncha), groups=filters_per_branch),
                nn.BatchNorm2d(filters_per_branch *2),
                getattr(nn, activation)(),
                nn.Dropout(dropout_rate)
        ]

        # Combine layers
        self.block1 = nn.Sequential(*layer1)
        self.block2 = nn.Sequential(*layer2)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x.squeeze_(-1)
        #print(x.shape)
        return x
    
class Conv1DNet(nn.Module):
    def __init__(self):
        super(Conv1DNet, self).__init__()
        # First 1D convolutional layer
        self.dropout = 0.1
        self.conv1 =  nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=64, stride = 1, padding = 'same'),
            #nn.BatchNorm1d(16),
            nn.LayerNorm([16, 512]),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        # # Second 1D convolutional layer
        self.conv2 =  nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=32, stride = 1, padding = 'same'),
            #nn.BatchNorm1d(32),
            nn.LayerNorm([32, 512]),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=128, kernel_size=16, stride =1,  padding = 'same'),
            #nn.BatchNorm1d(64),
            nn.LayerNorm([128, 512]),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

    def forward(self, x):
        # x -> (N, C_{in}, L_{in}
        # Apply first convolution, ReLU activation function, and max pooling
        x  = x.permute(0,2,1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x
 
class TSTransformerEncoder(nn.Module):

    def __init__(self, feat_dim, seq_len, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1,
                 pos_encoding='fixed', activation='gelu', norm='LayerNorm', freeze=False):
        super(TSTransformerEncoder, self).__init__()

        self.max_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads

        #self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout, max_len=seq_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout, activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout, activation=activation)

        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)

        #self.output_layer = nn.Linear(d_model, feat_dim)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim

    def forward(self, X):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """
        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = X.permute(2,0,1)  # (seq_length, batch_size, feat_dim)
        #inp = self.project_inp(inp) * math.sqrt(self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        #inp = inp.permute(0,2,1)  # (batch_size, seq_length, d_model)
        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output, attn  = self.transformer_encoder(inp)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1,0,2)  # (batch_size, seq_length, d_model)
        #output = self.dropout1(output)
        # Most probably defining a Linear(d_model,feat_dim) vectorizes the operation over (seq_length, batch_size).
        #output = self.output_layer(output)  # (batch_size, seq_length, feat_dim)
        return output, attn
    

