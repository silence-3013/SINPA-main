import torch
import numpy as np
import os
import time
import argparse
import yaml
import pickle
import scipy.sparse as sp
from scipy.sparse import linalg

import torch.nn as nn
import torch

import sys
import os

cur_path = os.path.abspath(os.path.dirname(__file__))  # 当前脚本目录
project_root = os.path.abspath(os.path.join(cur_path, "../../"))  # 项目根目录

sys.path.insert(0, project_root)  # 将项目根加入模块搜索路径


from src.utils.helper import get_dataloader, check_device, get_num_nodes
from src.models.DeepPA import DeepPA
from src.trainers.deeppa_trainer import DeepPA_Trainer
from src.utils.graph_algo import load_graph_data
from src.utils.args import get_public_config, str_to_bool


def get_config():
    parser = get_public_config()

    # get private config
    parser.add_argument("--model_name", type=str, default="DeepPA", help="which model to train")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--filter_type", type=str, default="transition")
    parser.add_argument("--n_blocks", type=int, default=2)
    parser.add_argument("--n_hidden", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--spatial_flag", type=str_to_bool, default=True, help="whether to use spatial transformer")
    parser.add_argument("--temporal_flag", type=str_to_bool, default=True, help="whether to use temporal transformer")
    parser.add_argument("--spatial_encoding", type=str_to_bool, default=True, help="whether to use spatial encoding")
    parser.add_argument("--temporal_encoding", type=str_to_bool, default=True, help="whether to use temporal encoding")
    parser.add_argument("--temporal_PE", type=str_to_bool, default=True, help="whether to use temporal PE")
    parser.add_argument("--GCO", type=str_to_bool, default=True, help="whether to use GCO")
    parser.add_argument("--CLUSTER", type=str_to_bool, default=False, help="whether to use CLUSTER")
    parser.add_argument("--GCO_Thre", type=float, default=1, help="The proportion of low frequency signals")
    # GCO module improvements
    parser.add_argument("--gco_impl", type=str, default="fourier", choices=["fourier", "wavelet"]) 
    parser.add_argument("--gco_adaptive", type=str_to_bool, default=True)
    parser.add_argument("--gco_alpha", type=float, default=10.0)
    parser.add_argument("--gco_tau", type=float, default=0.0)
    parser.add_argument("--gco_wavelet_levels", type=int, default=1)
    parser.add_argument("--base_lr", type=float, default=1e-3)
    parser.add_argument("--lr_decay_ratio", type=float, default=0.5)
    args = parser.parse_args()
    args.steps = [10, 20, 30, 40]
    print(args)

    folder_name = "{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}".format(
        args.n_hidden,
        args.n_blocks,
        args.n_heads,
        args.spatial_flag,
        args.temporal_flag,
        args.spatial_encoding,
        args.temporal_encoding,
        args.temporal_PE,
        args.aug,
        args.batch_size,
        args.base_lr,
        args.n_exp,
        args.GCO,
        args.temporal_encoding,
        args.GCO_Thre,
    )
    args.log_dir = "./logs/{}/{}/{}/".format(args.dataset, args.model_name, folder_name)
    print(args.log_dir)
    args.num_nodes = get_num_nodes(args.dataset)

    args.datapath = os.path.join("./dataset", args.dataset)
    args.graph_pkl = "data/sensor_graph/adj_mx_{}.pkl".format(args.dataset.lower())
    if args.seed != 0:
        torch.manual_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    return args, folder_name


def main():
    print("【断点】程序启动，准备获取配置参数")  # 中文注释：主流程入口
    args, fname = get_config()
    print("【断点】配置参数：", args)  # 中文注释：打印所有参数

    device = check_device()
    print("【断点】使用设备：", device)  # 中文注释：设备信息
    _, _, adj_mat = load_graph_data(args.graph_pkl)
    print("【断点】图结构加载完成")  # 中文注释：图结构

    model = DeepPA(
        dropout=args.dropout,
        spatial_flag=args.spatial_flag,
        temporal_flag=args.temporal_flag,
        spatial_encoding=args.spatial_encoding,
        temporal_encoding=args.temporal_encoding,
        temporal_PE=args.temporal_PE,
        GCO=args.GCO,
        CLUSTER=args.CLUSTER,
        n_hidden=args.n_hidden,
        end_channels=args.n_hidden * 8,
        n_blocks=args.n_blocks,
        name=args.model_name,
        dataset=args.dataset,
        device=device,
        num_nodes=args.num_nodes,
        seq_len=args.seq_len,
        horizon=args.horizon,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        GCO_Thre=args.GCO_Thre,
        gco_impl=args.gco_impl,
        gco_adaptive=args.gco_adaptive,
        gco_alpha=args.gco_alpha,
        gco_tau=args.gco_tau,
        gco_wavelet_levels=args.gco_wavelet_levels,
    )
    print("【断点】DeepPA模型创建完成")  # 中文注释：模型初始化

    data = get_dataloader(args.datapath, args.batch_size, args.output_dim)
    print("【断点】数据加载器创建完成，keys:", data.keys())  # 中文注释：dataloader信息

    trainer = DeepPA_Trainer(
        model=model,
        adj_mat=adj_mat,
        filter_type=args.filter_type,
        data=data,
        aug=args.aug,
        base_lr=args.base_lr,
        steps=args.steps,
        lr_decay_ratio=args.lr_decay_ratio,
        log_dir=args.log_dir,
        n_exp=args.n_exp,
        wandb_flag=args.wandb,
        wandb_mode=args.wandb_mode,
        save_iter=args.save_iter,
        clip_grad_value=args.max_grad_norm,
        max_epochs=args.max_epochs,
        patience=args.patience,
        device=device,
    )
    print("【断点】Trainer创建完成")  # 中文注释：Trainer初始化

    if args.mode == "train":
        print("【断点】进入训练模式")  # 中文注释：训练模式
        trainer.train()
        trainer.test(-1, "test")
    else:
        print("【断点】进入测试模式")  # 中文注释：测试模式
        trainer.test(-1, args.mode)
        if args.save_preds:
            print("【断点】保存预测结果")  # 中文注释：保存预测
            trainer.save_preds(-1)


if __name__ == "__main__":
    torch.set_num_threads(8)
    main()
