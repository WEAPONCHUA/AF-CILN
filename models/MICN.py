import torch
import torch.nn as nn
from layers.Embed import DataEmbedding
from layers.Autoformer_EncDec import series_decomp, series_decomp_multi
import torch.nn.functional as F

"""
    Paper link: https://openreview.net/pdf?id=zt53IDUR1U
"""

number_heads = 1
d_layer = 1
embedding_size = 64


class MIC(nn.Module):
    """
    MIC layer to extract local and global features
    """

    def __init__(
        self,
        feature_size=128,
        n_heads=8,
        dropout=0.05,
        decomp_kernel=[32],
        conv_kernel=[24],
        isometric_kernel=[18, 6],
        device="cuda",
    ):
        super(MIC, self).__init__()
        self.conv_kernel = conv_kernel
        self.device = device

        # isometric convolution
        self.isometric_conv = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=feature_size,
                    out_channels=feature_size,
                    kernel_size=i,
                    padding=0,
                    stride=1,
                )
                for i in isometric_kernel
            ]
        )

        # downsampling convolution: padding=i//2, stride=i
        self.conv = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=feature_size,
                    out_channels=feature_size,
                    kernel_size=i,
                    padding=i // 2,
                    stride=i,
                )
                for i in conv_kernel
            ]
        )

        # upsampling convolution
        self.conv_trans = nn.ModuleList(
            [
                nn.ConvTranspose1d(
                    in_channels=feature_size,
                    out_channels=feature_size,
                    kernel_size=i,
                    padding=0,
                    stride=i,
                )
                for i in conv_kernel
            ]
        )

        self.decomp = nn.ModuleList([series_decomp(k) for k in decomp_kernel])
        self.merge = torch.nn.Conv2d(
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=(len(self.conv_kernel), 1),
        )

        # feedforward network
        self.conv1 = nn.Conv1d(
            in_channels=feature_size, out_channels=feature_size * 4, kernel_size=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=feature_size * 4, out_channels=feature_size, kernel_size=1
        )
        self.norm1 = nn.LayerNorm(feature_size)
        self.norm2 = nn.LayerNorm(feature_size)

        self.norm = torch.nn.LayerNorm(feature_size)
        self.act = torch.nn.Tanh()
        self.drop = torch.nn.Dropout(0.05)

    def conv_trans_conv(self, input, conv1d, conv1d_trans, isometric):
        batch, seq_len, channel = input.shape
        x = input.permute(0, 2, 1)

        # downsampling convolution
        x1 = self.drop(self.act(conv1d(x)))
        x = x1

        # isometric convolution
        zeros = torch.zeros(
            (x.shape[0], x.shape[1], x.shape[2] - 1), device=self.device
        )
        x = torch.cat((zeros, x), dim=-1)
        x = self.drop(self.act(isometric(x)))
        x = self.norm((x + x1).permute(0, 2, 1)).permute(0, 2, 1)

        # upsampling convolution
        x = self.drop(self.act(conv1d_trans(x)))
        x = x[:, :, :seq_len]  # truncate

        x = self.norm(x.permute(0, 2, 1) + input)
        return x

    def forward(self, src):
        self.device = src.device
        # multi-scale
        multi = []
        for i in range(len(self.conv_kernel)):
            src_out, trend1 = self.decomp[i](src)
            src_out = self.conv_trans_conv(
                src_out, self.conv[i], self.conv_trans[i], self.isometric_conv[i]
            )
            multi.append(src_out)

        # merge
        mg = torch.tensor([], device=self.device)
        for i in range(len(self.conv_kernel)):
            mg = torch.cat((mg, multi[i].unsqueeze(1).to(self.device)), dim=1)
        mg = self.merge(mg.permute(0, 3, 1, 2)).squeeze(-2).permute(0, 2, 1)

        y = self.norm1(mg)
        y = self.conv2(self.conv1(y.transpose(-1, 1))).transpose(-1, 1)

        return self.norm2(mg + y)


class SeasonalPrediction(nn.Module):
    def __init__(
        self,
        embedding_size=512,
        n_heads=8,
        dropout=0.05,
        d_layers=1,
        decomp_kernel=[32],
        c_out=1,
        conv_kernel=[2, 4],
        isometric_kernel=[18, 6],
        device="cuda",
    ):
        super(SeasonalPrediction, self).__init__()

        self.mic = nn.ModuleList(
            [
                MIC(
                    feature_size=embedding_size,
                    n_heads=n_heads,
                    decomp_kernel=decomp_kernel,
                    conv_kernel=conv_kernel,
                    isometric_kernel=isometric_kernel,
                    device=device,
                )
                for i in range(d_layers)
            ]
        )

        self.projection = nn.Linear(embedding_size, c_out)

    def forward(self, dec):
        for mic_layer in self.mic:
            dec = mic_layer(dec)
        return self.projection(dec)


class MICN(nn.Module):
    def __init__(self, configs, conv_kernel=[12, 16]):
        """
        conv_kernel: downsampling and upsampling convolution kernel_size
        """
        super(MICN, self).__init__()

        decomp_kernel = []  # kernel of decomposition operation
        isometric_kernel = []  # kernel of isometric convolution
        for ii in conv_kernel:
            if ii % 2 == 0:  # the kernel of decomposition operation must be odd
                decomp_kernel.append(ii + 1)
                isometric_kernel.append((configs.seq_len + configs.pred_len + ii) // ii)
            else:
                decomp_kernel.append(ii)
                isometric_kernel.append(
                    (configs.seq_len + configs.pred_len + ii - 1) // ii
                )

        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len

        # Multiple Series decomposition block from FEDformer
        self.decomp_multi = series_decomp_multi(decomp_kernel)

        # embedding
        self.dec_embedding = DataEmbedding(
            configs.enc_in,
            embedding_size,
            configs.embed,
            configs.freq,
            configs.dropout,
        )

        self.conv_trans = SeasonalPrediction(
            embedding_size=embedding_size,
            n_heads=number_heads,
            dropout=configs.dropout,
            d_layers=d_layer,
            decomp_kernel=decomp_kernel,
            c_out=configs.c_out,
            conv_kernel=conv_kernel,
            isometric_kernel=isometric_kernel,
            device=torch.device("cuda:0"),
        )
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            # refer to DLinear
            self.regression = nn.Linear(configs.seq_len, configs.pred_len)
            self.regression.weight = nn.Parameter(
                (1 / configs.pred_len)
                * torch.ones([configs.pred_len, configs.seq_len]),
                requires_grad=True,
            )
        if self.task_name == "imputation":
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forecast(self, x_enc, mask):
        # Multi-scale Hybrid Decomposition
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

        seasonal_init_enc, trend = self.decomp_multi(x_enc)
        trend = self.regression(trend.permute(0, 2, 1)).permute(0, 2, 1)

        # embedding
        zeros = torch.zeros(
            [mask.shape[0], self.pred_len, mask.shape[2]], device=x_enc.device
        )
        seasonal_init_dec = torch.cat(
            [seasonal_init_enc[:, -self.seq_len :, :], zeros], dim=1
        )
        dec_out = self.dec_embedding(seasonal_init_dec, None)
        dec_out = self.conv_trans(dec_out)
        dec_out = dec_out[:, -self.pred_len :, :] + trend[:, -self.pred_len :, :]

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, mask):
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            dec_out = self.forecast(x_enc, mask)
            return dec_out[:, -self.pred_len :, :3]  # [B, L, D]
        return None
