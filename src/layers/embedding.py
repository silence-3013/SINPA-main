import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    """
    A module that performs time embedding for input data.

    Args:
        None

    Attributes:
        embed_slot (nn.Embedding): Embedding layer for slot information.
        embed_day (nn.Embedding): Embedding layer for day of the week information.
        embed_util (nn.Embedding): Embedding layer for utilization type information.
        embed_plan (nn.Embedding): Embedding layer for geolocation information.

    Methods:
        forward(x): Performs the forward pass of the time embedding layer.

    """

    def __init__(self):
        super(TimeEmbedding, self).__init__()
        # a day contains 96 15-min slots
        self.embed_slot = nn.Embedding(4 * 24, 3)
        self.embed_day = nn.Embedding(7, 3)  # Day of the Week
        self.embed_util = nn.Embedding(10, 3)  # Utilization Type
        self.embed_plan = nn.Embedding(36, 3)  # Geolocation

    def forward(self, x):
        """
        Performs the forward pass of the time embedding layer.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, time_steps, num_features].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, time_steps, num_features_embedded].

        """
        # x: slot, day, holiday or not
        x_slot = self.embed_slot(x[..., 0])
        x_day = self.embed_day(x[..., 1])
        x_util = self.embed_util(x[..., 6])
        x_plan = self.embed_plan(x[..., 7])
        out = torch.cat((x_slot, x_day, x[..., 2:6], x_util, x_plan, x[..., 8:]), -1)
        return out  # [b, t, n, 3+3+4+3+3+3=19]
    
    """
1. 张量维度的基本概念
问题核心：代码中反复提到的 “维度” 指什么？
解答：在神经网络中，“维度” 指张量（多维数组）的轴（axis），每个维度代表数据的组织方向（如样本数、时间步、特征数等）。例如 3 维张量 [[[1],[2]],[[3],[4]]] 的形状 [2,2,1] 中，3 个维度分别对应 “样本数、时间步、特征数”，每个维度的数值代表该方向上的数据数量。
如果把这个 3 维张量类比为代码中处理的时序数据（如 [batch_size, time_steps, num_features]），可以这样对应：

假设这个张量表示 “不同批次的样本在多个时间步的特征数据”：

第一个维度（长度 2）→ batch_size=2
代表 2 个独立样本（批次大小为 2），即 [[1],[2]] 是样本 1，[[3],[4]] 是样本 2。
第二个维度（长度 2）→ time_steps=2
每个样本包含 2 个时间步，例如样本 1 的时间步 1 是 [1]，时间步 2 是 [2]。
第三个维度（长度 1）→ num_features=1
每个时间步只有 1 个特征（例如温度），所以时间步 1 的特征是 1，时间步 2 的特征是 2。
2. 输入张量 X 的维度含义
问题核心：[b, t, n, input_dim] 各字母代表什么？具体数值（如 [8,12,1687,12]）的含义？
解答：
b（batch size）：批次大小（如 8，表示一次输入 8 个样本）；
t（time steps）：时间步长（如 12，表示每个样本包含 12 个时间点）；
n（nodes）：节点数量（如 1687，表示每个时间步有 1687 个空间节点）；
input_dim：输入特征维度（如 12，表示每个节点在每个时间步有 12 维特征）。
3. 嵌入操作的基本逻辑
问题核心：x_embed = self.embed(X[..., 1:].long()) 中，输入和输出的维度变化（如从 [8,12,1687,12] 到 [8,12,1687,19]）是什么意思？
解答：
输入 X[..., 1:] 表示提取除第 0 维外的 11 维特征（形状 [8,12,1687,11]），.long() 转换为整数类型（嵌入层要求输入为离散整数）；
嵌入层将 11 维离散特征映射为 19 维连续向量（embed_dim=19），前三个维度（b,t,n）不变，仅特征维度变化。
4. TimeEmbedding 类的结构与嵌入层定义
问题核心：TimeEmbedding 类中定义的嵌入层（如 self.embed_slot = nn.Embedding(4*24, 3)）是什么意思？
解答：
嵌入层用于将离散整数特征（如时隙、星期几）转换为连续向量；
nn.Embedding(num_embeddings, embedding_dim) 中，num_embeddings 是特征类别数（如 4*24=96 表示一天的 15 分钟时隙数），embedding_dim=3 表示嵌入后向量的维度（3 维）。
5. forward 方法的特征提取逻辑
问题核心：forward 方法中如何提取和处理不同特征（如 x_slot = self.embed_slot(x[..., 0]) 是什么意思）？
解答：
通过 x[..., i] 提取特征维度（最后一个维度）的第 i 个特征（如 x[...,0] 是时隙特征，x[...,1] 是星期几特征）；
对离散特征（时隙、星期几等）用嵌入层转换为 3 维向量，对连续特征（如 x[...,2:6]）直接保留原始值。
6. forward 方法的特征拼接逻辑
问题核心：torch.cat(...) 如何拼接特征，最终维度 19 是如何计算的？
解答：
沿特征维度（最后一个维度）拼接所有处理后的特征：x_slot（3 维）+ x_day（3 维）+ x[...,2:6]（4 维）+ x_util（3 维）+ x_plan（3 维）+ x[...,8:]（3 维）；
总和为 3+3+4+3+3+3=19，最终输出形状为 [batch_size, time_steps, 19]。

一、为什么 __init__ 方法不定义输入形状？
__init__ 方法的核心作用是定义网络的可学习参数（如嵌入层的权重矩阵），而不是指定输入数据的具体形状。这是 PyTorch 等深度学习框架的通用设计原则，原因有两点：

网络层的参数与输入形状无关
以代码中的嵌入层为例：
python
运行
self.embed_slot = nn.Embedding(4*24, 3)  # 96个类别，3维嵌入


它的参数是一个固定形状的矩阵 [96, 3]（96 行对应 96 个时隙，3 列是嵌入维度），这个矩阵的大小只由 “类别数” 和 “嵌入维度” 决定，与输入数据有多少个样本（batch_size）、多少个时间步（time_steps）无关。
无论输入是 [8, 12, ...] 还是 [16, 24, ...]，嵌入层的参数矩阵都不会改变，因此 __init__ 无需关心输入形状。
"""
