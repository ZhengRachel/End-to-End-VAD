import torch
import torch.nn as nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, input_channel=1):
        super(Encoder, self).__init__()
        #[B, C, T, H, W]
        #[B, 1, T, 64, 128]
        
        dropout_p = 0.2

        self.convblock1 = nn.Sequential(
                        nn.Conv3d(input_channel, 32, (5, 5, 5), 
                            stride=(1, 2, 2),
                            padding=(2, 2, 2)), #[B, 32, T, 32, 64]
                        nn.BatchNorm3d(32),
                        nn.ReLU(),
                        nn.Dropout3d(p=dropout_p),
                        nn.MaxPool3d((1, 2, 2), #[B, 32, T, 16, 32]
                            stride=(1, 2, 2),
                            padding=0))
        self.convblock2 = nn.Sequential(
                            nn.Conv3d(32, 64, (5, 5, 5), 
                                stride=(1, 2, 2),
                                padding=(2, 2, 2)), #[B, 64, T, 8, 16]
                            nn.BatchNorm3d(64),
                            nn.ReLU(),
                            nn.Dropout3d(p=dropout_p),
                            nn.MaxPool3d((1, 2, 2), #[B, 64, T, 4, 8]
                                stride=(1, 2, 2),
                                padding=0))
        self.convblock3 = nn.Sequential(
                        nn.Conv3d(64, 128, (5, 3, 3), 
                            stride=(1, 2, 2),
                            padding=(2, 1, 1)), #[B, 128, T, 2, 4]
                        nn.BatchNorm3d(128),
                        nn.ReLU(),
                        nn.Dropout3d(p=dropout_p))
        self.project = nn.Linear(1024, 512)
        self.activate = nn.Sequential(nn.BatchNorm1d(512),
                       nn.ReLU(),
                       nn.Dropout(p=dropout_p))
        

    def forward(self, x):
        '''
        x [B, c, T, H, W]

        o [B, T, 512]
        '''
        x = x.unsqueeze(1) #-> [B, 1, T, 64, 128]
        # print(x.shape)
        c1 = self.convblock1(x) # [B, 32, T, 16, 32]
        c2 = self.convblock2(c1) # [B, 64, T, 4, 8]
        c3 = self.convblock3(c2) # [B, 128, T, 2, 4]

        o = c3.permute(0, 2, 1, 3, 4) # [B, T, 128, 2, 4]
        o = o.reshape(o.size(0), -1, 1024) # [B, T, 1024]

        o = self.project(o)
        o = o.transpose(1, 2) # -> [B, 512, T]
        o = self.activate(o)
        o = o.transpose(1, 2) # -> [B, T, 512] 
        
        return o
