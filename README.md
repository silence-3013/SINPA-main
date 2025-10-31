# SINPA / DeepPA 运行与验证指南
本仓库实现了我们在 IJCAI 2024（AI for Social Good Track）的工作：[Predicting Carpark Availability in Singapore with Cross-Domain Data: A New Dataset and A Data-Driven Approach](https://arxiv.org/pdf/2405.18910)。
我们爬取并发布了大规模跨域停车可用性数据集 **SINPA**，并提出了端到端预测框架 **DeepPA** 用于集体预测新加坡范围内的未来停车可用性（PA）。

## Framework
<img src="img/intro and model.png" width="1000px">

Figure (a) Distribution of 1,687 carparks throughout Singapore. (b) The framework of DeepPA.

## 快速开始（Windows/PowerShell）

### 1) 创建环境并安装依赖

建议使用 Python 3.9。Torch 版本已在 `requirements.txt` 固定为 1.13.0。

```
conda create -n sinpa python=3.9 -y
conda activate sinpa
pip install -r requirements.txt
# 由于代码中全局导入了 wandb，建议安装：
pip install wandb
# 如需使用一键下载脚本下载 Hugging Face 上的数据文件：
pip install huggingface_hub
```

如果你不使用 conda，直接用系统 Python 也可以，仅需按上面 pip 安装。

### 2) 准备数据

- 基础数据集默认使用 `--dataset base`，对应目录结构如下（仓库已包含 demo 数据）：

```
dataset/
  └── base/
      ├── train.npz
      ├── val.npz
      └── test.npz
```

- 图结构与辅助文件位于 `data/`（用于度量与支持矩阵计算），至少需要：

```
data/
  ├── sensor_graph/adj_mx_base.pkl   # 图的邻接矩阵（已提供或可自动下载）
  ├── prob_full_occupy.npy           # 评估掩码相关文件
  └── region/{assignment.npy, mask.npy}
```

- 一键下载上述 `data/` 文件（可选）：

```
python .\download_data.py
```

- 如果你只是想快速跑通代码而没有图结构，可以生成一个“单位矩阵邻接”（开发调试用，不建议用于评价指标）：

```
python .\create_fake_adj.py
```

### 3) 设备与日志

- 代码会自动检测 GPU/CPU。`--gpu` 仅用于设置 `CUDA_VISIBLE_DEVICES`，CPU 也能运行但较慢。
- 日志与模型输出位于：`logs/<dataset>/<model_name>/<auto-folder>/`。

## 数据集简介
我们从 [Data.gov.sg](https://data.gov.sg/) 爬取了三年、每 5 分钟的实时 PA 数据，覆盖 1,921 个停车场。为减轻缺失值影响，我们将原始数据重采样为 15 分钟，并选取缺失率 <30% 的停车场。考虑时间分布漂移，仅使用 2020/07/01 – 2021/06/30 的一年数据，训练/验证/测试比例为 10:1:1。我们同时移除 KL 散度较高的停车场样本，最终保留 1,687 个分布相对稳定的停车场。此外，我们整合了气象、规划区、利用类型、路网等外部属性，来源包括 [Data.gov.sg](https://data.gov.sg/)、[URA](https://www.ura.gov.sg/)、[LTA](https://datamall.lta.gov.sg/content/datamall/en.html)。

下表为各特征维度说明：

<table>
  <capital></capital>
  <tr>
  <th>Dimension</th>
  <th>Type</th>
  <th>Category</th>
  <th>Feature name</th>
  <th>Detail</th>
  </tr>
  <tr>
  <td >0</td>
  <td >Predict Target</td>
  <td >Parking Availability</td>
  <td >Parking Availability</td>
  <td >Real value</td>
  </tr>
  <tr>
  <td >1</td>
  <td rowspan=6>Temporal Factor<br></td>
  <td rowspan=3>Time-related<br></td>
  <td >Time of day</td>
  <td >0 to 95 int number (24*4)</td>
  </tr>
  <tr>
  <td >2</td>
  <td >Weekday</td>
  <td >0 to 6 int number (7)</td>
  </tr>
  <tr>
  <td >3</td>
  <td >Is_holiday</td>
  <td >One-hot</td>
  </tr>
  <td >4</td>
  <td rowspan=3>Meteorology<br></td>
  <td >Temperature</td>
  <td >Normalized value</td>
  </tr>
  <td >5</td>
  <td >Humidity</td>
  <td >Normalized value</td>
  </tr>
  </tr>
  <td >6</td>
  <td >Windspeed</td>
  <td >Normalized value</td>
  </tr>
  <td >7</td>
  <td rowspan=5>Spatial Factor<br></td>
  <td >Utilization Type</td>
  <td >Utilization Type</td>
  <td >0 to 9 int number (10)</td>
  </tr>
  <td >8</td>
  <td >Region-related</td>
  <td >Planning area</td>
  <td >0 to 35 int number (36)</td>
  </tr>
  </tr>
  <td >9</td>
  <td >Road-related</td>
  <td >Road Density</td>
  <td >Normalized value</td>
  </tr>
  <td >10</td>
  <td rowspan=2>Location<br></td>
  <td >Latitude</td>
  <td >Normalized value</td>
  </tr>
  </tr>
  <td >11</td>
  <td >Longitude</td>
  <td >Normalized value</td>
  </tr>
  </table>

Note: _Normalized_ refers to Z-score normalization, which is applied for fast convergence.

辅助数据：如果需要可视化停车场或自定义邻接矩阵，可使用 `aux_data/lots_location.csv`。

## 运行示例（含 GCO 改进）
以下示例基于 `--dataset base` 与默认超参（`n_hidden=64, n_blocks=2, n_heads=2, batch_size=8` 等）。Windows PowerShell 请使用 `python .\experiments\DeepPA\main.py`，Linux/Mac 可使用 `python ./experiments/DeepPA/main.py`。

### 1) 基线（原始傅里叶 GCO，非自适应）

```
python .\experiments\DeepPA\main.py --dataset base --mode train --gpu 0 \
  --GCO True --gco_impl fourier --gco_adaptive False --GCO_Thre 1 \
  --wandb True --wandb_mode offline
```

### 2) 傅里叶 + 自适应频域门控

```
python .\experiments\DeepPA\main.py --dataset base --mode train --gpu 0 \
  --GCO True --gco_impl fourier --gco_adaptive True \
  --gco_alpha 10.0 --gco_tau 0.0 --wandb True --wandb_mode offline
```

### 3) 小波（Haar） + 自适应频域门控

```
python .\experiments\DeepPA\main.py --dataset base --mode train --gpu 0 \
  --GCO True --gco_impl wavelet --gco_adaptive True \
  --gco_alpha 10.0 --gco_tau 0.0 --gco_wavelet_levels 1 \
  --wandb True --wandb_mode offline
```

### 4) 关闭 GCO（对比）

```
python .\experiments\DeepPA\main.py --dataset base --mode train --gpu 0 \
  --GCO False --wandb True --wandb_mode offline
```

### 5) 调整低频比例（非自适应）

```
python .\experiments\DeepPA\main.py --dataset base --mode train --gpu 0 \
  --GCO True --gco_impl fourier --gco_adaptive False --GCO_Thre 0.7 \
  --wandb True --wandb_mode offline
```

## 测试与保存预测

- 测试（会加载训练时保存的最佳权重，并输出 MAE/RMSE）：

```
python .\experiments\DeepPA\main.py --dataset base --mode test --gpu 0 \
  --wandb True --wandb_mode offline
```

- 保存预测（可选）：

```
python .\experiments\DeepPA\main.py --dataset base --mode test --gpu 0 \
  --save_preds True --wandb True --wandb_mode offline
```

输出与日志位置示例：

- 模型：`logs/base/DeepPA/<自动文件夹>/final_model_<n_exp>.pt`
- 验证与测试指标：终端打印，平均 MAE/RMSE 写入 `results_<n_exp>.csv`
- 预测保存：`train_preds.npy / val_preds.npy / test_preds.npy`（对应标签文件亦保存）

## 新增/关键参数释义（GCO 改进）

- `--gco_impl {fourier,wavelet}`：选择频域实现（傅里叶或 Haar 小波）。
- `--gco_adaptive {True,False}`：是否启用自适应频域门控；启用后不再使用 `GCO_Thre` 的硬阈值。
- `--gco_alpha <float>`：门控陡峭度（越大越接近硬掩码），默认 10.0。
- `--gco_tau <float>`：门控阈值（能量中心阈或频率阈的平移量），默认 0.0。
- `--gco_wavelet_levels <int>`：Haar 小波分解层级，当前实现支持 1。
- `--GCO_Thre <float>`：非自适应下低频保留比例（0~1），默认 1。

其他常用训练参数：
- `--batch_size`（默认 8，显存不足时可调小）
- `--max_epochs`（默认 100）与 `--patience`（默认 10，早停）
- `--wandb` 与 `--wandb_mode`（如不使用请设为 `--wandb False --wandb_mode disabled`）
 - `--wandb` 与 `--wandb_mode`（推荐使用 `--wandb True --wandb_mode offline` 以离线记录）

## 实验配置（原论文）与复现建议

- 环境与设备：原论文使用 `PyTorch 1.10` 与 `Quadro RTX A6000` GPU；当前仓库在 `torch==1.13.0` 下已验证可运行，若需完全对齐可使用 `1.10`。
- 优化器：`Adam`，初始学习率 `1e-3`（对应 `--base_lr 1e-3`）。
- 批大小：`--batch_size 8`。
- 学习率调度：原论文为“每 3 个 epoch 减半”。当前代码默认使用 `MultiStepLR(milestones=[10,20,30,40], gamma=0.5)`。
  - 如需对齐原设，修改 `experiments/DeepPA/main.py` 中的里程碑为：
    - `args.steps = list(range(3, args.max_epochs + 1, 3))`
  - 或自行设置一个较短的训练轮数并将里程碑设为其倍数。
- 隐藏维度 C（SLBlock/TLBlock）：在 `{8,16,32,64,128}` 网格搜索，最佳为 `C=64`（对应 `--n_hidden 64`）。
- 模块数量：`SLBlock` 与 `TLBlock` 均为 2（对应 `--n_blocks 2`）。
- 注意：`--n_heads` 默认 `2`，与原设一致。

复现命令（对齐原论文主要超参）：

```
python .\experiments\DeepPA\main.py --dataset base --mode train --gpu 0 \
  --n_hidden 64 --n_blocks 2 --n_heads 2 --batch_size 8 \
  --base_lr 1e-3 --wandb True --wandb_mode offline
```

如需完全对齐“每 3 个 epoch 减半”，请按上述方式调整 `args.steps`。

## 目录结构
主要模块代码位置：

1. 训练与测试入口：`experiments/DeepPA/main.py`
2. 模型实现：`src/models/DeepPA.py`
3. 训练与评估：`src/trainers/deeppa_trainer.py`、`src/base/trainer.py`
4. 数据加载与工具：`src/utils/*`

## 常见问题（Troubleshooting）

- 运行时报错 `ModuleNotFoundError: No module named 'wandb'`
  - 解决：`pip install wandb`，或在命令行添加 `--wandb False --wandb_mode disabled`（注意：代码中有全局导入，建议安装）。
- 评估时提示缺少 `prob_full_occupy.npy` 或 `region/mask.npy`
  - 解决：执行 `python .\download_data.py` 下载到 `data/`，或手动放置对应文件。
- 缺少图的邻接 `data/sensor_graph/adj_mx_base.pkl`
  - 解决：执行 `python .\download_data.py`，或用 `python .\create_fake_adj.py` 生成一个占位版本（仅用于跑通）。
- 显存不足（CUDA OOM）
  - 解决：减小 `--batch_size`；或切到 CPU（速度较慢）。

## License
The <b>SINPA</b> dataset is released under the Singapore Open Data Licence: [https://beta.data.gov.sg/open-data-license](https://beta.data.gov.sg/open-data-license).

## Citation
If you find our work useful in your research, please cite:
```
@inproceedings{zhang2024predicting,
  title={Predicting Parking Availability in Singapore with Cross-Domain Data: A New Dataset and A Data-Driven Approach},
  author={Zhang, Huaiwu and Xia, Yutong and Zhong, Siru and Wang, Kun and Tong, Zekun and Wen, Qingsong and Zimmermann, Roger and Liang, Yuxuan},
  booktitle={Proceedings of the Thirty-third International Joint Conference on Artificial Intelligence, IJCAI-24},
  year={2024}
}
```