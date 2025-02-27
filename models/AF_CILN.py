import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


def haar_transform(x):
    """Apply the Haar wavelet transform on the last dimension of x."""
    n = x.size(-1)
    if n == 1:
        return x

    # Ensure that the length of the signal is a power of 2
    assert (n & (n - 1) == 0) and n != 0, "Input size must be a power of 2"

    x_even = x[..., ::2]
    x_odd = x[..., 1::2]

    h = (x_even + x_odd) / np.sqrt(2)  # Approximation coefficients
    g = (x_even - x_odd) / np.sqrt(2)  # Detail coefficients

    if n > 2:
        # Recursively apply the transform to the approximation coefficients
        h = haar_transform(h)

    # Concatenate approximation and detail coefficients
    transformed = torch.cat([h, g], dim=-1)

    return transformed


def adaptive_haar_transform(x):
    """
    Apply the Adaptive Haar wavelet transform on the last dimension of x.

    Args:
        x (torch.Tensor): Input tensor, where the last dimension is transformed.

    Returns:
        torch.Tensor: Transformed tensor with approximation and detail coefficients concatenated.
    """
    n = x.size(-1)
    if n == 1:
        return x

    # Decompose into even and odd indexed parts
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]

    # Compute approximation and detail coefficients
    h = (x_even + x_odd) / np.sqrt(2)
    g = (x_even - x_odd) / np.sqrt(2)

    if n > 2:
        if n % 2 == 0:
            # Recursively apply the transform to the approximation coefficients
            h = adaptive_haar_transform(h)
        else:
            # Handle odd length by removing the last element and treating it separately
            last_element = x[..., -1:]
            x_trimmed = x[..., :-1]
            h_trimmed = adaptive_haar_transform(
                (x_trimmed[..., ::2] + x_trimmed[..., 1::2]) / np.sqrt(2)
            )
            g_trimmed = (x_trimmed[..., ::2] - x_trimmed[..., 1::2]) / np.sqrt(2)
            h = h_trimmed
            g = torch.cat([g_trimmed, last_element], dim=-1)

    # Concatenate approximation and detail coefficients
    transformed = torch.cat([h, g], dim=-1)
    return transformed


class haar_channel_block(nn.Module):
    def __init__(self, channel):
        super(haar_channel_block, self).__init__()
        self.channel = channel
        self.fc = nn.Sequential(
            nn.Linear(channel, channel * 2, bias=False),
            nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),
            nn.Linear(channel * 2, channel, bias=False),
            nn.Sigmoid(),
        )
        self.haar_norm = nn.LayerNorm([channel], eps=1e-6)

    def forward(self, x):
        b, l, c = x.size()  # (B, L, D)

        list = []
        for i in range(c):
            freq = haar_transform(x[:, :, i])
            list.append(freq)

        stack_haar = torch.stack(list, dim=1)

        lr_weight = self.haar_norm(stack_haar)
        lr_weight = self.fc(stack_haar)
        lr_weight = self.haar_norm(lr_weight)

        lr_weight = lr_weight.permute(0, 2, 1)
        return x * lr_weight


class Splitting(nn.Module):
    def __init__(self):
        super(Splitting, self).__init__()

    def even(self, x):
        return x[:, ::2, :]

    def odd(self, x):
        return x[:, 1::2, :]

    def forward(self, x):
        # return the odd and even part
        return self.even(x), self.odd(x)


class CausalConvBlock(nn.Module):
    def __init__(self, d_model, kernel_size=5, dropout=0.0):
        super(CausalConvBlock, self).__init__()
        module_list = [
            nn.ReplicationPad1d((kernel_size - 1, kernel_size - 1)),
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size),
            nn.Tanh(),
        ]
        self.causal_conv = nn.Sequential(*module_list)

    def forward(self, x):
        return self.causal_conv(x)  # return value is the same as input dimension


class Convolutional_Interactive_Learning_Block(nn.Module):
    def __init__(self, d_model, kernel_size=5, dropout=0.0):
        super(Convolutional_Interactive_Learning_Block, self).__init__()
        self.splitting = Splitting()
        (
            self.modules_even,
            self.modules_odd,
            self.interactor_even,
            self.interactor_odd,
        ) = [CausalConvBlock(d_model) for _ in range(4)]

    def forward(self, x):
        x_even, x_odd = self.splitting(x)
        x_even = x_even.permute(0, 2, 1)
        x_odd = x_odd.permute(0, 2, 1)

        x_even_temp = x_even.mul(torch.exp(self.modules_even(x_odd)))
        x_odd_temp = x_odd.mul(torch.exp(self.modules_odd(x_even)))

        x_even_update = x_even_temp + self.interactor_even(x_odd_temp)
        x_odd_update = x_odd_temp - self.interactor_odd(x_even_temp)

        return x_even_update.permute(0, 2, 1), x_odd_update.permute(0, 2, 1)


class CILN(nn.Module):
    def __init__(self, d_model, current_level=3, kernel_size=5, dropout=0.0):
        super(CILN, self).__init__()
        self.current_level = current_level
        self.working_block = Convolutional_Interactive_Learning_Block(
            d_model, kernel_size, dropout
        )

        if current_level != 0:
            self.SCINet_Tree_odd = CILN(
                d_model, current_level - 1, kernel_size, dropout
            )
            self.SCINet_Tree_even = CILN(
                d_model, current_level - 1, kernel_size, dropout
            )

    def forward(self, x):
        odd_flag = False
        if x.shape[1] % 2 == 1:
            odd_flag = True
            x = torch.cat((x, x[:, -1:, :]), dim=1)
        x_even_update, x_odd_update = self.working_block(x)
        if odd_flag:
            x_odd_update = x_odd_update[:, :-1]

        if self.current_level == 0:
            return self.zip_up_the_pants(x_even_update, x_odd_update)
        else:
            return self.zip_up_the_pants(
                self.SCINet_Tree_even(x_even_update), self.SCINet_Tree_odd(x_odd_update)
            )

    def zip_up_the_pants(self, even, odd):
        even = even.permute(1, 0, 2)
        odd = odd.permute(1, 0, 2)
        even_len = even.shape[0]
        odd_len = odd.shape[0]
        min_len = min(even_len, odd_len)

        zipped_data = []
        for i in range(min_len):
            zipped_data.append(even[i].unsqueeze(0))
            zipped_data.append(odd[i].unsqueeze(0))
        if even_len > odd_len:
            zipped_data.append(even[-1].unsqueeze(0))
        return torch.cat(zipped_data, 0).permute(1, 0, 2)


class AF_CILN(nn.Module):
    def __init__(self, configs):
        super(AF_CILN, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.hidden_dim = configs.d_model
        self.res_hidden = configs.d_model

        self.haar_channel_block = haar_channel_block(self.seq_len)

        # numb_stacks = 1
        self.num_stacks = 1
        if self.num_stacks == 1:
            self.cil_net_1 = CILN(configs.enc_in, dropout=configs.dropout)
            self.projection_1 = nn.Conv1d(
                self.seq_len,
                self.seq_len + self.pred_len,
                kernel_size=1,
                stride=1,
                bias=False,
            )
        # For positional encoding
        self.pe_hidden_size = configs.enc_in
        if self.pe_hidden_size % 2 == 1:
            self.pe_hidden_size += 1

        num_timescales = self.pe_hidden_size // 2
        max_timescale = 10000.0
        min_timescale = 1.0

        log_timescale_increment = math.log(
            float(max_timescale) / float(min_timescale)
        ) / max(num_timescales - 1, 1)
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) * -log_timescale_increment
        )
        self.register_buffer("inv_timescales", inv_timescales)

    def forward(self, x_enc, mask=None):
        dec_out = self.forecast(x_enc, mask)  # [B,pred_len,C]
        dec_out = torch.cat([torch.zeros_like(x_enc), dec_out], dim=1)
        return dec_out[:, -self.pred_len :, :3]  # [B, T, D]

    def forecast(self, x_enc, mask):
        # 标准化
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

        # AWHT
        x_enc = self.haar_channel_block(x_enc)

        # position-encoding
        pe = self.get_position_encoding(x_enc)
        if pe.shape[2] > x_enc.shape[2]:
            x_enc += pe[:, :, :-1]
        else:
            x_enc += self.get_position_encoding(x_enc)

        # CILN
        dec_out = self.cil_net_1(x_enc)
        dec_out += x_enc
        dec_out = self.projection_1(dec_out)
        if self.num_stacks != 1:
            dec_out = torch.cat((x_enc, dec_out), dim=1)
            temp = dec_out
            dec_out = self.cil_net_2(dec_out)
            dec_out += temp
            dec_out = self.projection_2(dec_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (
            stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len + self.pred_len, 1)
        )
        dec_out = dec_out + (
            means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len + self.pred_len, 1)
        )
        return dec_out

    def get_position_encoding(self, x):
        max_length = x.size()[1]
        position = torch.arange(
            max_length, dtype=torch.float32, device=x.device
        )  # tensor([0., 1., 2., 3., 4.], device='cuda:0')
        scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0)  # 5 256
        signal = torch.cat(
            [torch.sin(scaled_time), torch.cos(scaled_time)], dim=1
        )  # [T, C]
        signal = F.pad(signal, (0, 0, 0, self.pe_hidden_size % 2))
        signal = signal.view(1, max_length, self.pe_hidden_size)

        return signal
