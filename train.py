import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import argparse
import models
from utils import *
from dataloader import MyDataset
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_percentage_error,
    mean_absolute_error,
)  # MSE,  MAPE, MAE
from clearml import Task  # Visualization
from clearml import Logger  # Visualization
import math

# Create a result folder
if not os.path.exists("result"):
    os.makedirs("result")


# Loss function considering mask
class TrajectoryCompletionLoss(nn.Module):
    def __init__(self):
        super(TrajectoryCompletionLoss, self).__init__()

    def forward(self, predicted_traj, true_traj, mask):
        # Calculate mean squared error loss
        mse_loss = nn.MSELoss(reduction="none")(predicted_traj, true_traj)

        # Only consider the loss of the complete data part
        masked_mse_loss = mse_loss * mask

        # Calculate the average loss
        loss = torch.sum(masked_mse_loss) / torch.sum(mask)

        return loss


def load_dataset(data_file_name, label_file_name):
    dataset = MyDataset(
        f"dataset/{data_file_name}.npy",
        f"dataset/{label_file_name}.npy",
    )
    return dataset


# KNN replacement function
def replace_with_avg(arr, ratio):
    mask = np.random.rand(*arr.shape[:2]) < ratio
    masked_arr = np.copy(arr)
    masked_arr[mask] = np.nan

    for i in range(2, arr.shape[1] - 2):
        masked_arr[:, i] = np.nanmean(arr[:, i - 2 : i + 3], axis=1)

    return masked_arr


def main(args):

    # ClearML visual record
    task = Task.init(
        project_name="HGV_imputation",
        task_name=f"{args.model_name}-i{args.seq_len}-o{args.pred_len}-r{args.missing_radio}-T{args.task_type}",
    )
    logger = Logger.current_logger()

    dataset = load_dataset(args.data_file_name, args.label_file_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ratio = args.train_ratio
    num_samples = len(dataset)
    num_train = int(train_ratio * num_samples)
    num_test = num_samples - num_train
    train_dataset, test_dataset = random_split(dataset, [num_train, num_test])
    if args.model_name == "MICN":
        model = models.MICN(configs=args)
    elif args.model_name == "LightTS":
        model = models.LightTS(args=args)
    elif args.model_name == "CILN":
        model = models.CILN_2(configs=args)
    elif args.model_name == "AF_CILN":
        model = models.AF_CILN(configs=args)
    elif args.model_name == "iTransformer":
        model = models.iTransformer(args=args)
    elif args.model_name == "SS_LSTM":
        model = models.LSTM(args=args)
    else:
        model = models.AF_CILN(configs=args)
    model.to(device)

    mask_function = create_symmetric_mask
    mask_single = mask_function(
        (args.batch_size, args.seq_len, args.enc_in), args.missing_radio
    ).to(device)

    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Print model information
    print(model)
    params = list(model.parameters())
    print(params[0].size())

    optimizer = torch.optim.AdamW(params, lr=args.learning_rate)
    loss_function = nn.MSELoss().to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer, step_size=10, gamma=0.5
    )
    time_start = time.time()

    num_epochs = args.num_epochs

    # train
    for epoch in range(num_epochs):

        model.train()
        train_loss = 0.0

        for batch in train_loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            B, L, D = inputs.size()
            mask_new = mask_single[:B, :, :]
            inputs = inputs.float()
            optimizer.zero_grad()
            outputs = model(inputs, mask_new)
            loss = loss_function(outputs, labels.float())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        logger.report_scalar(
            title="Train", series="loss", value=train_loss, iteration=epoch
        )

        # validate
        model.eval()
        test_loss = 0.0
        test_mape = 0.0
        test_mae = 0.0
        test_mse = 0.0
        test_r2 = 0.0

        with torch.no_grad():
            for batch in test_loader:
                inputs, labels = batch[0].to(device), batch[1].to(device)
                inputs = inputs.float()
                B, L, D = inputs.size()
                mask_new = mask_single[:B, :, :]
                outputs = model(inputs, mask_new)
                loss = loss_function(outputs, labels.float())
                test_loss += loss.item()

                # In order to facilitate the calculation of various indicators, convert the output to (batch_size, -1)
                labels = (labels).reshape(labels.shape[0], -1)
                outputs = (outputs).reshape(outputs.shape[0], -1)

                # Calculate R2
                r2 = r2_score(labels.cpu().numpy(), outputs.cpu().numpy())
                test_r2 += r2

                # Calculate MAPE
                mape = mean_absolute_percentage_error(
                    labels.cpu().numpy(), outputs.cpu().numpy()
                )
                test_mape += mape

                # Calculate MAE
                mae = mean_absolute_error(labels.cpu().numpy(), outputs.cpu().numpy())
                test_mae += mae

                # Calculate MSE
                mse = mean_squared_error(labels.cpu().numpy(), outputs.cpu().numpy())
                test_mse += mse

        test_loss /= len(test_loader)
        test_r2 /= len(test_loader)
        test_mape /= len(test_loader)
        test_mae /= len(test_loader)
        test_mse /= len(test_loader)

        logger.report_scalar(
            title="test_loss", series="loss", value=test_loss, iteration=epoch
        )
        logger.report_scalar(
            title="test_mape", series="mape(%)", value=test_mape, iteration=epoch
        )
        logger.report_scalar(
            title="test_mae", series="mae", value=test_mae, iteration=epoch
        )
        logger.report_scalar(
            title="test_rmse", series="rmse", value=test_mse**0.5, iteration=epoch
        )
        logger.report_scalar(
            title="test_r2", series="r2", value=test_r2, iteration=epoch
        )
        # Print results
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test r2:{test_r2:.4f}:, Test Loss: {test_loss:.4f},Test MAPE: {test_mape*100:.4f}%,Test MAE: {test_mae:.4f},Test RMSE: {test_mse**0.5:.4f}"
        )

    # Save the last round of model as model name + timestamp + pt
    current_timestamp = str(int(time.time()))
    save_path = os.path.join(
        "result",
        f"{args.model_name}-i{args.seq_len}-o{args.pred_len}--{args.task_type}-{args.missing_radio}-{current_timestamp}.pt",
    )
    model.cpu()
    torch.save(model, save_path)
    total_time = time.time() - time_start
    print(f"total time is {total_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="AF_CILN")
    parser.add_argument(
        "--task_name",
        type=str,
        default="long_term_forecast",
    )
    parser.add_argument(
        "--data_file_name",
        type=str,
        default="data_i256_o256_small",
    )
    parser.add_argument(
        "--label_file_name",
        type=str,
        default="label_i256_o256_small",
    )
    parser.add_argument(
        "--seq_len", type=int, default=256, help="Input sequence length"
    )
    parser.add_argument(
        "--pred_len", type=int, default=256, help="Predicting sequence length"
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="C",
        help="Trajectory interpolation type, including R (random missing) and C (continuous missing)",
    )
    parser.add_argument(
        "--missing_radio", type=float, default=0, help="Missing proportion"
    )
    parser.add_argument(
        "--enc_in", type=int, default=6, help="Number of input channels"
    )
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--seg_len", type=int, default=16)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--e_layers", type=int, default=3)
    parser.add_argument("--d_layers", type=int, default=3)
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--freq", type=str, default="s")
    # Other parameters, such as rounds, batch_size, learning rate, etc.
    parser.add_argument("--num_epochs", type=int, default=400)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=0.0002)
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    args = parser.parse_args()
    print(args)
    main(args)
    torch.cuda.empty_cache()
