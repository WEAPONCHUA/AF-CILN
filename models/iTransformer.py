import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np
from layers.RevIN import RevIN

"""
    Paper link: https://arxiv.org/abs/2310.06625
"""


# fixed
output_attention = False
embed = "timeF"
freq = "s"
factor = 1
# Basic parameters
c_in = 6  # Number of input channels
c_out = 3  # Number of output channels
# Tunable parameters
e_layers = 4  # Number of encoders
d_model = 512  # Dimensions of linear layers in encoder and decoder
dropout = 0.3
n_heads = 16  # Number of attention heads
d_ff = 2048  # Fully connected layer dimensions
activation = "gelu"  # Activation function type


class iTransformer(nn.Module):

    def __init__(self, args):
        super(iTransformer, self).__init__()
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.output_attention = output_attention
        self.revin_layer = RevIN(c_in)
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(
            self.seq_len,
            d_model,
            embed,
            freq,
            dropout,
        )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )
        # Decoder
        self.projection = nn.Linear(d_model, self.pred_len, bias=True)

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
        _, _, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, mask):
        dec_out = self.forecast(x_enc, mask)
        return dec_out[:, -self.pred_len :, :c_out]  # [B, L, D]
