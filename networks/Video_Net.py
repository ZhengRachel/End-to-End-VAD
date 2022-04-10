import torch
import torch.nn as nn
from networks.Encoder import Encoder

# VIDEO only network
class DeepVAD_video(nn.Module):

    def __init__(self, device):
        super(DeepVAD_video, self).__init__()
        self.device = device
        self.build_model()

    def build_model(self):
        self.encoder = Encoder(input_channel=1)

        self.lstm_video = nn.LSTM(input_size=512,
                            hidden_size=512,
                            num_layers=2,
                            bidirectional=False)

        self.vad_video = nn.Linear(512, 2)
        
        self.dropout = nn.Dropout(p=0.5)

    def parse_batch(self, batch):
        input, target = batch
        input = input.to(self.device).float()
        target = target.to(self.device).long()
        return (input, target)

    def forward(self, input):
        # print(input.size())
        encoder_out = self.encoder(input) # output shape - [B, T, 512]
        # Reshape to (T, B, 512)
        encoder_out = encoder_out.transpose(0, 1)
        # output shape - seq len X Batch X lstm size
        lstm_out, _ = self.lstm_video(encoder_out)  
        # select last time step. many -> one
        lstm_out = self.dropout(lstm_out[-1])  
        out = torch.sigmoid(self.vad_video(lstm_out))
        return out