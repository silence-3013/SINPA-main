# from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import DataLoader, TensorDataset
from src.utils.data import Dataset
import torch
from torch import Tensor
import logging
import numpy as np
import os
import sys
import pickle

from src.utils.scaler import StandardScaler


def get_dataloader(datapath, batch_size, output_dim, mode="train"):
    print("【断点】开始加载数据，路径：", datapath)  # 中文注释：数据加载入口
    data = {}
    processed = {}
    results = {}
    
    # Load data from files
    for category in ["train", "val", "test"]:
        print(os.path.join(datapath, category + ".npz"))
        cat_data = np.load(os.path.join(datapath, category + ".npz"), allow_pickle=True)
        data["x_" + category] = cat_data["x"]
        data["y_" + category] = cat_data["y"]
        print(f"【断点】加载 {category} 数据: x shape {data['x_' + category].shape}, y shape {data['y_' + category].shape}")  # 中文注释：每个数据集的shape
        print(data["x_" + category].shape)
        print(data["y_" + category].shape)

    scalers = []
    for i in range(output_dim):  
        # Create scalers for standardization based on the training data
        scalers.append(
            StandardScaler(
                mean=data["x_train"][..., i].mean(), std=data["x_train"][..., i].std()
            )
        )

    # Normalize the data
    for category in ["train", "val", "test"]:
        for i in range(output_dim):
            data["x_" + category][..., i] = scalers[i].transform(
                data["x_" + category][..., i]
            )
            data["y_" + category][..., i] = scalers[i].transform(
                data["y_" + category][..., i]
            )
        print(f"【断点】{category} 归一化后 x[0]：", data["x_" + category][0])  # 中文注释：归一化后样本
        print(f"【断点】{category} 归一化后 y[0]：", data["y_" + category][0])

        new_x = Tensor(data["x_" + category])
        new_y = Tensor(data["y_" + category])
        processed[category] = TensorDataset(new_x, new_y)
        print(category)
        print(new_x.shape)
        print(new_y.shape)

    # Create dataloaders
    results["train_loader"] = DataLoader(processed["train"], batch_size, shuffle=True)
    results["val_loader"] = DataLoader(processed["val"], batch_size, shuffle=False)
    results["test_loader"] = DataLoader(processed["test"], batch_size, shuffle=False)

    print("【断点】train_loader size:", len(results["train_loader"].dataset))  # 中文注释：训练集样本数
    print(
        "train: {}\t valid: {}\t test:{}".format(
            len(results["train_loader"].dataset),
            len(results["val_loader"].dataset),
            len(results["test_loader"].dataset),
        )
    )
    results["scalers"] = scalers
    return results


def check_device(device=None):
    """
    Checks and returns the device to be used for training and evaluation.

    Args:
        device (torch.device or str, optional): The device to be used. If not provided, the default device will be used.

    Returns:
        torch.device: The device to be used for training and evaluation.

    Raises:
        TypeError: If the provided device is not of type torch.device or str.

    """
    if device is None:
        print("`device` is missing, try to train and evaluate the model on default device.")
        if torch.cuda.is_available():
            print("cuda device is available, place the model on the device.")
            return torch.device("cuda")
        else:
            print("cuda device is not available, place the model on cpu.")
            return torch.device("cpu")
    else:
        if isinstance(device, torch.device):
            return device
        else:
            return torch.device(device)


def get_num_nodes(dataset):
    """
    Get the number of nodes for a given dataset.

    Parameters:
    dataset (str): The name of the dataset.

    Returns:
    int: The number of nodes for the given dataset.

    Raises:
    AssertionError: If the dataset is not found in the dictionary.

    """
    d = {"base": 1687, "base_new": 1697, "tiny": 1687}
    assert dataset in d.keys()
    return d[dataset]


def get_dataframe(datapath, batch_size, output_dim, mode="train"):
    """
    Load and process data from the specified datapath.

    Args:
        datapath (str): The path to the data directory.
        batch_size (int): The batch size for the data.
        output_dim (int): The output dimension of the data.
        mode (str, optional): The mode of the data. Defaults to "train".

    Returns:
        list: A list containing the processed training and testing data.
    """
    data = {}
    processed = {}
    results = {}
    for category in ["train", "test"]:
        print(os.path.join(datapath, category + ".npz"))
        cat_data = np.load(os.path.join(datapath, category + ".npz"), allow_pickle=True)
        data["x_" + category] = cat_data["x"]
        data["y_" + category] = cat_data["y"]
        print(data["x_" + category].shape)
        print(data["y_" + category].shape)
        if category == "train":
            train_ = np.squeeze(cat_data["y"][0:, 0:1, 0:, 0:])
        if category == "test":
            test_ = np.squeeze(cat_data["y"][0:, 0:1, 0:, 0:])
    return [train_, test_]
