import torch
import torch.nn as nn
import torch.nn.functional as F


# Basic parameters
channel_in = 6
channel_out = 3
# Tunable parameters
num_layers = 6
hidden_size = 64


class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.input_size = channel_in  # channel
        self.hidden_size = hidden_size  # Output dimension is also called output channel
        self.num_layers = num_layers
        self.output_size = channel_in  # channel
        self.num_directions = 1  # Unidirectional LSTM
        self.lstm = nn.LSTM(
            self.input_size, self.hidden_size, self.num_layers, batch_first=True
        )
        self.linear = nn.Linear(
            self.hidden_size, self.output_size
        )  # Channel Alignment Layer
        self.linear_out_len = nn.Linear(
            self.seq_len, self.pred_len
        )  # Output Length Alignment Layer

    def forecast(self, x_enc, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(
            torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5
        )
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev
        B, L, C = x_enc.size()
        h_0 = torch.randn(
            self.num_directions * self.num_layers,
            B,
            self.hidden_size,
            device=x_enc.device,
        )
        c_0 = torch.randn(
            self.num_directions * self.num_layers,
            B,
            self.hidden_size,
            device=x_enc.device,
        )

        x_enc, _ = self.lstm(x_enc, (h_0, c_0))
        x_enc = self.linear(x_enc)  # (B，L,C)
        x_enc = self.linear_out_len(x_enc.permute(0, 2, 1))
        dec_out = x_enc.permute(0, 2, 1)
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x, mask):
        x = self.forecast(x, mask)
        return x[:, -self.pred_len :, :channel_out]  # [B, L, D]
