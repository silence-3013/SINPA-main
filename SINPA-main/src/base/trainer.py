import logging
import os
import time
import json
from typing import Optional, List, Union

import numpy as np
import torch
from torch import nn, Tensor
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

from src.utils.logging import get_logger
from src.utils import metrics as mc
from src.utils.metrics import masked_mae
from src.base.sampler import RandomSampler, CutEdgeSampler

import pandas as pd
import wandb


class BaseTrainer:
    """
    Base class for trainers in the DeepPA project.

    Args:
        model (nn.Module): The neural network model.
        adj_mat: The adjacency matrix.
        filter_type (str): The type of filter.
        data: The training data.
        aug (float): The augmentation factor.
        base_lr (float): The base learning rate.
        steps: The steps for learning rate decay.
        lr_decay_ratio: The learning rate decay ratio.
        log_dir (str): The directory for logging.
        n_exp (int): The experiment number.
        wandb_flag (str, optional): Flag for using Weights & Biases logging. Defaults to True.
        save_iter (int, optional): The iteration interval for saving the model. Defaults to 300.
        clip_grad_value (Optional[float], optional): The maximum gradient value for gradient clipping. Defaults to None.
        max_epochs (Optional[int], optional): The maximum number of epochs. Defaults to 1000.
        patience (Optional[int], optional): The patience for early stopping. Defaults to 1000.
        device (Optional[Union[torch.device, str]], optional): The device for training and evaluation. Defaults to None.
    """

    def __init__(
        self,
        model: nn.Module,
        adj_mat,
        filter_type: str,
        data,
        aug: float,
        base_lr: float,
        steps,
        lr_decay_ratio,
        log_dir: str,
        n_exp: int,
        wandb_flag: str = True,
        save_iter: int = 300,
        clip_grad_value: Optional[float] = None,
        max_epochs: Optional[int] = 1000,
        patience: Optional[int] = 1000,
        device: Optional[Union[torch.device, str]] = None,
    ):
        super().__init__()

        self._logger = get_logger(
            log_dir, __name__, "info_{}.log".format(n_exp), level=logging.INFO
        )
        if device is None:
            print(
                "`device` is missing, try to train and evaluate the model on default device."
            )
            if torch.cuda.is_available():
                print("cuda device is available, place the model on the device.")
                self._device = torch.device("cuda")
            else:
                print("cuda device is not available, place the model on cpu.")
                self._device = torch.device("cpu")
        else:
            if isinstance(device, torch.device):
                self._device = device
            else:
                self._device = torch.device(device)

        self._model = model
        self._wandb_flag = wandb_flag
        self.model.to(self._device)
        self.num_param = self.model.param_num(self.model.name)
        self.nan_val = -1

        self._logger.info("the number of parameters: {}".format(self.num_param))
        if wandb_flag:
            wandb.init(project="DeepPA")
            wandb.run.summary["Params"] = self.num_param

        self._adj_mat = adj_mat
        self._filter_type = filter_type
        self._aug = aug
        self._loss_fn = masked_mae
        # self._loss_fn.to(self._device)
        self._base_lr = base_lr
        self._optimizer = Adam(self.model.parameters(), base_lr)
        self._lr_decay_ratio = lr_decay_ratio
        self._steps = steps
        if lr_decay_ratio == 1:
            self._lr_scheduler = None
        else:
            self._lr_scheduler = MultiStepLR(
                self.optimizer, steps, gamma=lr_decay_ratio
            )
        self._clip_grad_value = clip_grad_value
        self._max_epochs = max_epochs
        self._patience = patience
        self._save_iter = save_iter
        self._save_path = log_dir
        self._n_exp = n_exp
        self._data = data
        self._supports = None

        if aug > 0:
            self._sampler = RandomSampler(adj_mat, filter_type)

    @property
    def model(self):
        return self._model

    @property
    def supports(self):
        return self._supports

    @property
    def data(self):
        return self._data

    @property
    def logger(self):
        return self._logger

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def lr_scheduler(self):
        return self._lr_scheduler

    @property
    def loss_fn(self):
        return self._loss_fn

    @property
    def device(self):
        return self._device

    @property
    def save_path(self):
        return self._save_path

    def _check_device(self, tensors: Union[Tensor, List[Tensor]]):
        if isinstance(tensors, list):
            return [tensor.to(self._device) for tensor in tensors]
        else:
            return tensors.to(self._device)

    def _inverse_transform(self, tensors: Union[Tensor, List[Tensor]]):
        n_output_dim = 1

        def inv(tensor, scalers):
            for i in range(n_output_dim):
                tensor[..., i] = scalers[i].inverse_transform(tensor[..., i])
            return tensor

        if isinstance(tensors, list):
            return [inv(tensor, self.data["scalers"]) for tensor in tensors]
        else:
            return inv(tensors, self.data["scalers"])

    def _to_numpy(self, tensors: Union[Tensor, List[Tensor]]):
        if isinstance(tensors, list):
            return [tensor.cpu().detach().numpy() for tensor in tensors]
        else:
            return tensors.cpu().detach().numpy()

    def _to_tensor(self, nparray):
        if isinstance(nparray, list):
            return [Tensor(array) for array in nparray]
        else:
            return Tensor(nparray)

    def save_model(self, epoch, save_path, n_exp):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = "final_model_{}.pt".format(n_exp)
        torch.save(self.model.state_dict(), os.path.join(save_path, filename))
        return True

    def load_model(self, epoch, save_path, n_exp):
        filename = "final_model_{}.pt".format(n_exp)
        self.model.load_state_dict(torch.load(os.path.join(save_path, filename)))
        return True

    def early_stop(self, epoch, best_loss):
        self.logger.info(
            "Early stop at epoch {}, loss = {:.6f}".format(epoch, best_loss)
        )
        np.savetxt(
            os.path.join(self.save_path, "val_loss_{}.txt".format(self._n_exp)),
            [best_loss],
            fmt="%.4f",
            delimiter=",",
        )

    def _calculate_supports(self, adj_mat, filter_type):
        return None

    def train_batch(self, X, label, iter):
        # if self._aug < 1:
        #     new_adj = self._sampler.sample(self._aug)
        #     supports = self._calculate_supports(new_adj, self._filter_type)
        # else:
        # supports = self.supports
        self.optimizer.zero_grad()
        pred = self.model(X, self.supports)
        pred, label = self._inverse_transform([pred, label])

        loss = self.loss_fn(pred, label, self.nan_val)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self._clip_grad_value
        )
        self.optimizer.step()
        return loss.item()

    def train(self):
        self.logger.info("start training !!!!!")

        # training phase
        iter = 0
        val_losses = [np.inf]
        saved_epoch = -1
        for epoch in range(self._max_epochs):
            self.model.train()
            train_losses = []
            if epoch - saved_epoch > self._patience:
                if self._wandb_flag:
                    wandb.run.summary["final_valid_loss"] = min(val_losses)
                    wandb.run.summary["saved_epoch"] = saved_epoch
                self.early_stop(epoch, min(val_losses))
                break

            start_time = time.time()
            for i, (X, label) in enumerate(self.data["train_loader"]):
                X, label = self._check_device([X, label])
                train_losses.append(self.train_batch(X, label, iter))
                iter += 1
                if iter != None and self._save_iter > 0:
                    if iter % self._save_iter == 0:
                        val_loss = self.evaluate()
                        message = "Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f} ".format(
                            epoch,
                            self._max_epochs,
                            iter,
                            np.mean(train_losses),
                            val_loss,
                        )
                        self.logger.info(message)

                        if val_loss < np.min(val_losses):
                            model_file_name = self.save_model(
                                epoch, self._save_path, self._n_exp
                            )
                            self._logger.info(
                                "Val loss decrease from {:.4f} to {:.4f}, "
                                "saving to {}".format(
                                    np.min(val_losses), val_loss, model_file_name
                                )
                            )
                            val_losses.append(val_loss)
                            saved_epoch = epoch

            end_time = time.time()
            self.logger.info("epoch complete")
            self.logger.info("evaluating now!")

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            val_loss = self.evaluate()

            if self.lr_scheduler is None:
                new_lr = self._base_lr
            else:
                new_lr = self.lr_scheduler.get_last_lr()[0]

            message = (
                "Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f}, lr: {:.6f}, "
                "{:.1f}s".format(
                    epoch,
                    self._max_epochs,
                    iter,
                    np.mean(train_losses),
                    val_loss,
                    new_lr,
                    (end_time - start_time),
                )
            )
            self._logger.info(message)

            if val_loss < np.min(val_losses):
                model_file_name = self.save_model(epoch, self._save_path, self._n_exp)
                self._logger.info(
                    "Val loss decrease from {:.4f} to {:.4f}, "
                    "saving to {}".format(np.min(val_losses), val_loss, model_file_name)
                )
                val_losses.append(val_loss)
                saved_epoch = epoch

            if self._wandb_flag:
                wandb.log({"val_loss": val_loss, "epoch": epoch})
                wandb.log({"train_loss": np.mean(train_losses), "epoch": epoch})
                wandb.run.summary["final_valid_loss"] = min(val_losses)

    def evaluate(self):
        labels = []
        preds = []
        with torch.no_grad():
            self.model.eval()
            for _, (X, label) in enumerate(self.data["val_loader"]):
                X, label = self._check_device([X, label])
                pred, label = self.test_batch(X, label)
                labels.append(label.cpu())
                preds.append(pred.cpu())

        labels = torch.cat(labels, dim=0)
        preds = torch.cat(preds, dim=0)
        mae = self.loss_fn(preds, labels, self.nan_val).item()
        return mae

    def test_batch(self, X, label):
        pred = self.model(X, self.supports)
        pred, label = self._inverse_transform([pred, label])
        return pred, label

    def test(self, epoch, mode="test"):
        self.load_model(epoch, self.save_path, self._n_exp)

        labels = []
        preds = []
        with torch.no_grad():
            self.model.eval()
            for _, (X, label) in enumerate(self.data[mode + "_loader"]):
                X, label = self._check_device([X, label])
                pred, label = self.test_batch(X, label)
                labels.append(label.cpu())
                preds.append(pred.cpu())

        labels = torch.cat(labels, dim=0)
        preds = torch.cat(preds, dim=0)
        metrics = mc.compute_all_metrics(preds, labels, self.nan_val)
        log = "====Using evaluate function for test data -- Test MAE: {:.4f}, Test RMSE: {:.4f}===="
        print(log.format(metrics[0], metrics[1]))

        amae = []
        # aacc = []
        armse = []

        for i in range(self.model.horizon):
            pred = preds[:, i : i + 1]
            real = labels[:, i : i + 1]
            metrics = mc.compute_all_metrics(pred, real, self.nan_val)
            log = "====Using evaluate function for test data -- Test MAE: {:.4f}, Test RMSE: {:.4f}===="
            print(log.format(metrics[0], metrics[1]))
            amae.append(metrics[0])
            armse.append(metrics[1])

        log = "On average over {} horizons, Average Test MAE: {:.4f}, Test RMSE: {:.4f}"
        print(log.format(self.model.horizon, np.mean(amae), np.mean(armse)))

        if self._wandb_flag:
            wandb.run.summary["test MAE"] = np.mean(amae)
            wandb.run.summary["test RMSE"] = np.mean(armse)

        if self.model.horizon == 12:
            amae_day = []
            armse_day = []

            for i in range(0, self.model.horizon, 4):
                pred = preds[:, i : i + 4]
                real = labels[:, i : i + 4]
                metrics = mc.compute_all_metrics(pred, real, self.nan_val)
                # print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
                amae_day.append(metrics[0])
                armse_day.append(metrics[1])

            log = "0-3 (1h) Test MAE: {:.4f}, Test RMSE: {:.4f}"
            print(log.format(amae_day[0], armse_day[0]))
            log = "4-7 (2h) Test MAE: {:.4f}, Test RMSE: {:.4f}"
            print(log.format(amae_day[1], armse_day[1]))
            log = "8-11 (3h) Test MAE: {:.4f}, Test RMSE: {:.4f}"
            print(log.format(amae_day[2], armse_day[2]))

        results = np.stack([amae, armse], axis=0)
        np.savetxt(
            os.path.join(self.save_path, "results_{}.csv".format(self._n_exp)),
            results,
            fmt="%.4f",
            delimiter=",",
        )

        # --- Write structured metrics files with parameter info ---
        try:
            # collect params: prefer params.json, otherwise derive from model/trainer
            params = {}
            params_path = os.path.join(self.save_path, "params.json")
            if os.path.exists(params_path):
                try:
                    with open(params_path, "r", encoding="utf-8") as f:
                        params = json.load(f)
                except Exception:
                    params = {}
            # derive minimal params if missing
            if not params:
                # model-level
                params.update({
                    "model_name": getattr(self.model, "name", "DeepPA"),
                    "dataset": getattr(self.model, "dataset", None),
                    "num_nodes": getattr(self.model, "num_nodes", None),
                    "seq_len": getattr(self.model, "seq_len", None),
                    "horizon": getattr(self.model, "horizon", None),
                    "input_dim": getattr(self.model, "input_dim", None),
                    "output_dim": getattr(self.model, "output_dim", None),
                    # DeepPA specific
                    "dropout": getattr(self.model, "dropout", None),
                    "n_blocks": getattr(self.model, "n_blocks", None),
                    "n_hidden": getattr(self.model, "n_hidden", None),
                    "n_heads": getattr(self.model, "n_heads", None),
                    "spatial_flag": getattr(self.model, "spatial_flag", None),
                    "temporal_flag": getattr(self.model, "temporal_flag", None),
                    "spatial_encoding": getattr(self.model, "spatial_encoding", None),
                    "temporal_encoding": getattr(self.model, "temporal_encoding", None),
                    "temporal_PE": getattr(self.model, "temporal_PE", None),
                    "GCO": getattr(self.model, "GCO", None),
                    "CLUSTER": getattr(self.model, "CLUSTER", None),
                    "GCO_Thre": getattr(self.model, "GCO_Thre", None),
                })
                # trainer-level
                params.update({
                    "base_lr": getattr(self, "_base_lr", None),
                    "lr_decay_ratio": getattr(self, "_lr_decay_ratio", None),
                    "aug": getattr(self, "_aug", None),
                    "max_epochs": getattr(self, "_max_epochs", None),
                    "patience": getattr(self, "_patience", None),
                    "n_exp": getattr(self, "_n_exp", None),
                    "log_dir": getattr(self, "_save_path", None),
                })
                # batch size from loaders
                bs = None
                try:
                    for key in [f"{mode}_loader", "train_loader", "val_loader", "test_loader"]:
                        if key in self._data and hasattr(self._data[key], "batch_size"):
                            bs = self._data[key].batch_size
                            break
                except Exception:
                    bs = None
                params["batch_size"] = bs

            # assemble metrics payload
            metrics_dict = {
                "split": mode,
                "n_exp": self._n_exp,
                "horizon": self.model.horizon,
                "average": {"mae": float(np.mean(amae)), "rmse": float(np.mean(armse))},
                "per_horizon": {"mae": [float(x) for x in amae], "rmse": [float(x) for x in armse]},
                "log_dir": self.save_path,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "params": params,
            }

            # json
            json_path = os.path.join(self.save_path, f"metrics_{mode}_{self._n_exp}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metrics_dict, f, ensure_ascii=False, indent=2)

            # txt
            txt_path = os.path.join(self.save_path, f"metrics_{mode}_{self._n_exp}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("Params:\n")
                for k in [
                    "model_name","dataset","num_nodes","seq_len","horizon","input_dim","output_dim",
                    "dropout","n_blocks","n_hidden","n_heads","spatial_flag","temporal_flag",
                    "spatial_encoding","temporal_encoding","temporal_PE","GCO","CLUSTER","GCO_Thre",
                    "batch_size","base_lr","lr_decay_ratio","aug","max_epochs","patience","n_exp","log_dir"
                ]:
                    if k in params:
                        f.write(f"  - {k}: {params[k]}\n")
                f.write("\n")
                f.write(f"Average MAE: {metrics_dict['average']['mae']:.4f}\n")
                f.write(f"Average RMSE: {metrics_dict['average']['rmse']:.4f}\n")
                f.write("\nPer-horizon:\n")
                for h in range(self.model.horizon):
                    f.write(
                        f"  h={h+1}: MAE={amae[h]:.4f}, RMSE={armse[h]:.4f}\n"
                    )

            # csv with comment header
            csv_path = os.path.join(self.save_path, f"metrics_{mode}_{self._n_exp}.csv")
            with open(csv_path, "w", encoding="utf-8") as f:
                # compact param header line
                compact_pairs = []
                for k in [
                    "model_name","dataset","num_nodes","seq_len","horizon","input_dim","output_dim",
                    "dropout","n_blocks","n_hidden","n_heads","spatial_flag","temporal_flag",
                    "spatial_encoding","temporal_encoding","temporal_PE","GCO","CLUSTER","GCO_Thre",
                    "batch_size","base_lr","lr_decay_ratio","aug","max_epochs","patience","n_exp","log_dir"
                ]:
                    if k in params and params[k] is not None:
                        compact_pairs.append(f"{k}={params[k]}")
                f.write("# Params: " + ", ".join(compact_pairs) + "\n")
                f.write("h,mae,rmse\n")
                for h in range(self.model.horizon):
                    f.write(f"{h+1},{amae[h]:.4f},{armse[h]:.4f}\n")
        except Exception as e:
            # Do not break testing if metrics export fails
            self._logger.info(f"metrics export failed: {e}")

        # return amae, armse, aacc

    def save_preds(self, epoch):
        self.load_model(epoch, self.save_path, self._n_exp)

        for mode in ["train", "val", "test"]:
            labels = []
            preds = []
            inputs = []
            with torch.no_grad():
                self.model.eval()
                for _, (X, label) in enumerate(self.data[mode + "_loader"]):
                    X, label = self._check_device([X, label])
                    pred, label = self.test_batch(X, label)
                    labels.append(label.cpu())
                    preds.append(pred.cpu())
                    inputs.append(X.cpu())
            labels = torch.cat(labels, dim=0)
            preds = torch.cat(preds, dim=0)
            inputs = torch.cat(inputs, dim=0)
            np.save(os.path.join(self.save_path, mode + "_preds.npy"), preds)
            np.save(os.path.join(self.save_path, mode + "_labels.npy"), labels)
