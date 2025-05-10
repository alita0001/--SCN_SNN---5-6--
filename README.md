# 脉冲神经网络(SNN)并行训练框架

这个项目实现了一套基于脉冲神经网络(Spiking Neural Network, SNN)的训练框架，支持多种数据集上的分类和回归任务。项目结合了传统SCN(Stochastic Configuration Network)和SNN的优势，提供了高效的并行训练方法。

## 项目概述

本框架主要特点：

- 支持SCN与SNN的结合优化训练
- 提供多种脉冲神经元模型实现(LIF, IF, PLIF等)
- 支持在多个数据集上的并行训练与评估
- 包含完整的性能测试与可视化工具

## 核心组件

- `SCN_SNN.py`: 核心算法实现，包含SCN-SNN混合训练方法
- `snn_train.py`: 基于SpikingJelly的纯SNN训练实现
- `mnist_train.py`/`UCI_train.py`: 针对不同数据集的训练入口
- `image_process_copy.py`: 数据预处理模块
- 各种数据集专用训练脚本: `SCN_SNN_mnist.py`, `SCN_SNN_iris.py`等

## 支持的神经元模型

- LIF (Leaky Integrate-and-Fire)
- IF (Integrate-and-Fire)
- PLIF (Parametric Leaky Integrate-and-Fire)

## 数据集支持

- MNIST
- Fashion-MNIST
- UCI数据集(Iris, Wine, Breast Cancer, Ionosphere等)

## 使用方法

### 环境要求

- Python 3.6+
- PyTorch 1.7+
- NumPy
- SciPy
- Matplotlib
- tqdm
- SpikingJelly (可选，用于纯SNN训练)
- UMAP (用于数据降维)

### 安装依赖

```bash
pip install numpy scipy matplotlib tqdm torch umap-learn
pip install spikingjelly  # 可选
```

### 运行示例

1. **UCI数据集训练**:

```bash
python UCI_train.py
```

2. **MNIST数据集训练**:

```bash
python mnist_train.py
```

3. **模型评估**:

```bash
python UCI_val.py
```

## 并行训练特性

本框架支持多种并行训练方式:

1. 在脉冲神经网络中实现时间步并行计算
2. 批量数据的并行处理
3. 使用LIF层进行高效率计算

## 参考文献

- Cao, W., Wang, X., Ming, Z., & Gao, J. (2018). A review on neural networks with random weights. Neurocomputing, 275, 278-287.
- Zenke, F., & Ganguli, S. (2018). SuperSpike: Supervised learning in multilayer spiking neural networks. Neural computation, 30(6), 1514-1541.
- Wu, Y., Deng, L., Li, G., Zhu, J., & Shi, L. (2018). Spatio-temporal backpropagation for training high-performance spiking neural networks. Frontiers in neuroscience, 12, 331.

## 贡献

欢迎提交问题报告和拉取请求，共同改进这个项目。

## 许可证

MIT 