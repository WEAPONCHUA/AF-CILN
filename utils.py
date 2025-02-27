import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


# -------------------------- task_type = "C"
def create_symmetric_mask(shape, ratio):

    if not 0 <= ratio <= 1:
        raise ValueError("The scale value should be between [0, 1]")

    B, L, D = shape[0], shape[1], shape[2]

    mask = torch.ones(B, L, D)

    num_elements_per_trajectory = L * D
    num_zeros_per_trajectory = int(num_elements_per_trajectory * ratio)

    if num_zeros_per_trajectory > 0:
        half_zeros_per_trajectory = num_zeros_per_trajectory // 2

        for b in range(B):
            flat_mask = mask[b].view(-1)
            start_index = (num_elements_per_trajectory - num_zeros_per_trajectory) // 2
            for i in range(half_zeros_per_trajectory):
                flat_mask[start_index + i] = 0
                flat_mask[start_index + num_zeros_per_trajectory - 1 - i] = 0

    new_mask = mask.clone()

    for b in range(mask.size(0)):
        for l in range(mask.size(1)):
            row = mask[b, l]

            if 0 in row and 1 in row:
                new_mask[b, l] = 0

    return new_mask
