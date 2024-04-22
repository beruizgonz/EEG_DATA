import torch
from torch import nn

"""
Model of the EEG Inception architecture of the UVA university
"""    

class EEGInceptionBlock1(nn.Module):
    def __init__(self, in_channels, scales_samples, filters_per_branch, ncha, activation, dropout_rate):
        super(EEGInceptionBlock1, self).__init__()
        self.branches = nn.ModuleList()

        for scale_samples in scales_samples:
            layers = [
                nn.Conv2d(in_channels, filters_per_branch, kernel_size=(scale_samples, 1), padding='same'),
                nn.BatchNorm2d(filters_per_branch),
                getattr(nn, activation)(),
                nn.Dropout(dropout_rate),

                nn.Conv2d(filters_per_branch, filters_per_branch*2, kernel_size=(1, ncha), groups=filters_per_branch, bias=False),
                nn.BatchNorm2d(filters_per_branch *2),
                getattr(nn, activation)(),
                nn.Dropout(dropout_rate)
            ]
            self.branches.append(nn.Sequential(*layers))

    def forward(self, x):
        branch_outputs = [branch(x) for branch in self.branches]
        return torch.cat(branch_outputs, 1)
    

class EEGInceptionBlock2(nn.Module):
    def __init__(self, in_channels, scales_samples, filters_per_branch, ncha, activation, dropout_rate):
        super(EEGInceptionBlock2, self).__init__()
        self.branches = nn.ModuleList()

        for scale_samples in scales_samples:
            layers = [
                nn.Conv2d(in_channels, filters_per_branch, kernel_size=(scale_samples, 1), padding='same', bias = False),
                nn.BatchNorm2d(filters_per_branch),
                getattr(nn, activation)(),
                nn.Dropout(dropout_rate),
            ]
            self.branches.append(nn.Sequential(*layers))

    def forward(self, x):
        branch_outputs = [branch(x) for branch in self.branches]
        return torch.cat(branch_outputs, 1)


class EEGInception(nn.Module):
    def __init__(self, input_time=1000, fs=128, ncha=8, filters_per_branch=8, scales_time=(500, 250, 125), dropout_rate=0.25, activation='ELU', n_classes=2, learning_rate=0.001):
        super(EEGInception, self).__init__()
        input_samples = int(input_time * fs / 1000)
        scales_samples = [int(s * fs/ 1000) for s in scales_time]
        self.scales_samples = scales_samples
        self.block1 = EEGInceptionBlock1(1, scales_samples, filters_per_branch, ncha, activation, dropout_rate)
        self.avgpool1 = nn.AvgPool2d((4, 1))

        self.block2 = EEGInceptionBlock2(filters_per_branch * 2 * len(scales_samples), [int(s/4) for s in scales_samples], filters_per_branch, ncha, activation, dropout_rate)
        self.avgpool2 = nn.AvgPool2d((2, 1))

        self.conv3_1 = nn.Conv2d(int(filters_per_branch*len(scales_samples)), int(filters_per_branch*len(scales_samples)/2), kernel_size=(8, 1), padding='same', bias =False)
        self.bn3_1 = nn.BatchNorm2d(int(filters_per_branch * len(scales_samples) / 2))
        self.activation3_1 = getattr(nn, activation)()
        self.avgpool3_1 = nn.AvgPool2d((2, 1))
        self.dropout3_1 = nn.Dropout(dropout_rate)

        self.conv3_2 = nn.Conv2d(12, int(filters_per_branch*len(scales_samples)/4), kernel_size=(4, 1), padding='same', bias=False)
        self.bn3_2 = nn.BatchNorm2d(int(filters_per_branch * len(scales_samples) / 4))
        self.activation3_2 = getattr(nn, activation)()
        self.avgpool3_2 = nn.AvgPool2d((2, 1))
        self.dropout3_2 = nn.Dropout(dropout_rate)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(int(filters_per_branch*len(scales_samples)), n_classes) # Adjust the input features of fc layer according to your architecture
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        x = self.block1(x)

        x = self.avgpool1(x)
  
        x = self.block2(x)
        x = self.avgpool2(x)

        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.activation3_1(x)
        x = self.avgpool3_1(x)
        x = self.dropout3_1(x)
    
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = self.activation3_2(x)
        x = self.avgpool3_2(x)
        x = self.dropout3_2(x)
   
        x = self.flatten(x)
        x = self.fc(x)
        # x = self.sigmoid(x)
        x = x.squeeze(-1)
        return x
    