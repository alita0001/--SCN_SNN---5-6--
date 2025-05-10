#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于SpikingJelly搭建的脉冲神经网络(SNN)训练脚本
使用image_process_copy.py中的数据加载函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import os
from tqdm import tqdm

# 导入SpikingJelly相关库
from spikingjelly.activation_based import neuron, functional, surrogate, layer, monitor
from spikingjelly.activation_based.model import train_classify
from spikingjelly.activation_based.learning import STDPLearner

# 导入数据处理脚本
import image_process_copy as data_loader

# 设置随机种子，确保结果可复现
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class SpikingMLP(nn.Module):
    """
    基本的脉冲神经网络多层感知器
    
    参数:
    input_size: 输入特征维度
    hidden_size: 隐层神经元数量
    output_size: 输出类别数量
    n_timesteps: 时间步长
    neuron_type: 使用的神经元类型
    surrogate_function: 替代梯度函数
    detach_reset: 是否分离重置操作的梯度
    """
    def __init__(self, input_size, hidden_size, output_size, n_timesteps=8, 
                 neuron_type='LIF', surrogate_function=surrogate.ATan(), 
                 detach_reset=True):
        super(SpikingMLP, self).__init__()
        
        # 根据指定的神经元类型选择神经元
        if neuron_type == 'LIF':
            self.neuron = neuron.LIFNode
        elif neuron_type == 'IF':
            self.neuron = neuron.IFNode
        elif neuron_type == 'PLIF':
            self.neuron = neuron.ParametricLIFNode
        else:
            raise ValueError(f"不支持的神经元类型: {neuron_type}")
        
        # 构建网络层
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sn1 = self.neuron(surrogate_function=surrogate_function, detach_reset=detach_reset)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.sn2 = self.neuron(surrogate_function=surrogate_function, detach_reset=detach_reset)
        
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.sn_out = self.neuron(surrogate_function=surrogate_function, detach_reset=detach_reset)
        
        self.n_timesteps = n_timesteps
    
    def forward(self, x):
        # 将输入重复T次以形成时间序列
        if x.dim() == 2:
            x = x.unsqueeze(0).repeat(self.n_timesteps, 1, 1)  # [T, B, N]
        
        # 重置所有神经元的状态
        functional.reset_net(self)
        
        # 层间传播
        spikes = []
        for t in range(self.n_timesteps):
            x_t = x[t]
            
            x_t = self.fc1(x_t)
            x_t = self.sn1(x_t)
            
            x_t = self.fc2(x_t)
            x_t = self.sn2(x_t)
            
            x_t = self.fc_out(x_t)
            spike_out = self.sn_out(x_t)
            spikes.append(spike_out)
        
        # 沿时间维度堆叠输出
        return torch.stack(spikes, dim=0)  # [T, B, C]

class SpikingCNN(nn.Module):
    """
    基本的脉冲卷积神经网络(SCNN)
    
    适用于图像数据集，如MNIST, CIFAR10等
    """
    def __init__(self, input_channels, input_size, output_size, n_timesteps=8,
                 neuron_type='LIF', surrogate_function=surrogate.ATan(),
                 detach_reset=True):
        super(SpikingCNN, self).__init__()
        
        # 根据指定的神经元类型选择神经元
        if neuron_type == 'LIF':
            self.neuron = neuron.LIFNode
        elif neuron_type == 'IF':
            self.neuron = neuron.IFNode
        elif neuron_type == 'PLIF':
            self.neuron = neuron.ParametricLIFNode
        else:
            raise ValueError(f"不支持的神经元类型: {neuron_type}")
        
        # 计算图像展平后的尺寸
        img_width = int(np.sqrt(input_size / input_channels))
        
        # 构建网络
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.sn1 = self.neuron(surrogate_function=surrogate_function, detach_reset=detach_reset)
        
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.sn2 = self.neuron(surrogate_function=surrogate_function, detach_reset=detach_reset)
        
        self.pool2 = nn.MaxPool2d(2)
        
        # 计算展平后的特征维度
        flat_size = 32 * (img_width//4) * (img_width//4)
        
        self.fc1 = nn.Linear(flat_size, 256)
        self.sn3 = self.neuron(surrogate_function=surrogate_function, detach_reset=detach_reset)
        
        self.fc2 = nn.Linear(256, output_size)
        self.sn4 = self.neuron(surrogate_function=surrogate_function, detach_reset=detach_reset)
        
        self.n_timesteps = n_timesteps
        self.input_channels = input_channels
        self.img_width = img_width
    
    def forward(self, x):
        # 将输入重复T次以形成时间序列
        if x.dim() == 2:
            x = x.unsqueeze(0).repeat(self.n_timesteps, 1, 1)  # [T, B, N]
        
        # 重置所有神经元的状态
        functional.reset_net(self)
        
        # 根据输入形状重塑输入
        batch_size = x.shape[1]
        
        # 层间传播
        spikes = []
        for t in range(self.n_timesteps):
            # 当前时间步的输入
            x_t = x[t].view(batch_size, self.input_channels, self.img_width, self.img_width)
            
            # 通过第一层卷积
            x_t = self.conv1(x_t)
            x_t = self.sn1(x_t)
            x_t = self.pool1(x_t)
            
            # 通过第二层卷积
            x_t = self.conv2(x_t)
            x_t = self.sn2(x_t)
            x_t = self.pool2(x_t)
            
            # 展平
            x_t = x_t.flatten(1)
            
            # 通过全连接层
            x_t = self.fc1(x_t)
            x_t = self.sn3(x_t)
            
            x_t = self.fc2(x_t)
            spike_out = self.sn4(x_t)
            
            spikes.append(spike_out)
        
        # 沿时间维度堆叠输出
        return torch.stack(spikes, dim=0)  # [T, B, C]

def prepare_data(dataset_name, batch_size=128, use_cuda=True):
    """
    准备数据集
    
    参数:
    dataset_name: 数据集名称
    batch_size: 批次大小
    use_cuda: 是否使用CUDA
    
    返回:
    train_loader, test_loader: 训练和测试数据加载器
    """
    # 加载数据集
    print(f"加载{dataset_name}数据集...")
    
    try:
        # 从image_process_copy.py加载数据
        X_train, y_train, X_test, y_test = data_loader.load_dataset(dataset_name)
    except Exception as e:
        print(f"加载{dataset_name}失败: {e}")
        print("尝试更多参数...")
        
        # 为特殊数据集提供额外参数
        if dataset_name.lower() in ['cifar10', 'cifar100']:
            X_train, y_train, X_test, y_test = data_loader.load_dataset(dataset_name, grayscale=True)
        elif dataset_name.lower() == 'emnist':
            X_train, y_train, X_test, y_test = data_loader.load_dataset(dataset_name, split='balanced')
        else:
            raise ValueError(f"无法加载数据集: {dataset_name}")
    
    # 输出数据集形状信息
    print(f"训练集: {X_train.shape}, 标签: {y_train.shape}")
    print(f"测试集: {X_test.shape}, 标签: {y_test.shape}")
    
    # 转换为Tensor
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    
    # 创建数据集
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4 if use_cuda else 0,
        pin_memory=use_cuda,
        drop_last=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4 if use_cuda else 0,
        pin_memory=use_cuda,
        drop_last=False
    )
    
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, device, epochs=100, 
                lr=0.001, criterion=None, patience=10, save_path=None):
    """
    训练SNN模型
    
    参数:
    model: SNN模型
    train_loader: 训练数据加载器
    test_loader: 测试数据加载器
    device: 计算设备
    epochs: 训练周期数
    lr: 学习率
    criterion: 损失函数
    patience: 早停耐心值
    save_path: 模型保存路径
    
    返回:
    model: 训练好的模型
    train_losses: 训练损失
    test_accuracies: 测试准确率
    """
    model.to(device)
    if criterion is None:
        criterion = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    train_losses = []
    test_accuracies = []
    
    best_acc = 0
    no_improve_count = 0
    
    # 创建保存目录
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # 训练模式
        model.train()
        running_loss = 0.0
        
        for inputs, targets in tqdm(train_loader, desc=f"训练 Epoch {epoch+1}/{epochs}"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # 计算损失（对时间步骤求平均）
            loss = 0
            for t in range(outputs.shape[0]):
                loss += criterion(outputs[t], targets)
            loss = loss / outputs.shape[0]
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # 测试模式
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc=f"测试 Epoch {epoch+1}/{epochs}"):
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # 前向传播
                outputs = model(inputs)
                
                # 对时间步骤求平均
                outputs = outputs.mean(0)
                _, predicted = torch.max(outputs.data, 1)
                _, target_classes = torch.max(targets, 1)
                
                total += target_classes.size(0)
                correct += (predicted == target_classes).sum().item()
        
        acc = 100 * correct / total
        test_accuracies.append(acc)
        
        # 更新学习率
        scheduler.step()
        
        # 早停检查
        if acc > best_acc:
            best_acc = acc
            no_improve_count = 0
            
            # 保存最佳模型
            if save_path is not None:
                torch.save(model.state_dict(), save_path)
                print(f"保存模型到 {save_path}")
        else:
            no_improve_count += 1
        
        # 打印训练信息
        end_time = time.time()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Test Acc: {acc:.2f}%, "
              f"Time: {end_time-start_time:.2f}s, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # 如果连续patience个周期没有改善，则停止训练
        if no_improve_count >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # 加载最佳模型
    if save_path is not None and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
    
    return model, train_losses, test_accuracies

def plot_results(train_losses, test_accuracies, save_path=None):
    """
    绘制训练结果
    
    参数:
    train_losses: 训练损失
    test_accuracies: 测试准确率
    save_path: 图像保存路径
    """
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-')
    plt.title('训练损失')
    plt.xlabel('周期')
    plt.ylabel('损失')
    plt.grid(True)
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, 'r-')
    plt.title('测试准确率')
    plt.xlabel('周期')
    plt.ylabel('准确率 (%)')
    plt.grid(True)
    
    plt.tight_layout()
    
    # 保存图像
    if save_path is not None:
        plt.savefig(save_path)
        print(f"保存训练结果图到 {save_path}")
    
    plt.show()

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='SNN训练脚本')
    parser.add_argument('--dataset', type=str, default='mnist', help='数据集名称')
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'cnn'], help='模型类型')
    parser.add_argument('--hidden_size', type=int, default=256, help='隐层大小')
    parser.add_argument('--timesteps', type=int, default=8, help='时间步长')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练周期数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--neuron', type=str, default='LIF', choices=['LIF', 'IF', 'PLIF'], help='神经元类型')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='计算设备')
    parser.add_argument('--save_dir', type=str, default='./results', help='结果保存目录')
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 准备数据
    train_loader, test_loader = prepare_data(args.dataset, args.batch_size, use_cuda=(args.device=='cuda'))
    
    # 获取数据集信息
    input_size = None
    output_size = None
    
    # 从train_loader尝试获取一个批次
    try:
        for inputs, targets in train_loader:
            input_size = inputs.shape[1]
            output_size = targets.shape[1]
            break
    except:
        pass
    
    # 如果train_loader为空，则直接从数据集读取维度信息
    if input_size is None or output_size is None:
        try:
            # 重新加载数据但不使用DataLoader
            X_train, y_train, _, _ = data_loader.load_dataset(args.dataset)
            input_size = X_train.shape[1]
            output_size = y_train.shape[1]
            print(f"从原始数据获取维度信息")
        except Exception as e:
            print(f"无法获取数据维度信息: {e}")
            print("使用UCI Iris数据集的默认维度")
            # 默认维度，根据不同数据集设置
            if args.dataset.lower() == 'iris':
                input_size = 4
                output_size = 3
            elif args.dataset.lower() == 'breast_cancer':
                input_size = 30
                output_size = 2
            elif args.dataset.lower() in ['mnist', 'fmnist', 'kmnist']:
                input_size = 784  # 28*28
                output_size = 10
            elif args.dataset.lower() == 'cifar10':
                input_size = 1024  # 32*32 灰度图
                output_size = 10
            else:
                raise ValueError("无法自动确定数据维度，请手动指定")
    
    print(f"输入维度: {input_size}, 输出维度: {output_size}")
    
    # 确保batch_size不大于训练集大小
    actual_batch_size = min(args.batch_size, len(train_loader.dataset))
    if actual_batch_size != args.batch_size:
        print(f"警告: batch_size({args.batch_size})大于训练集大小({len(train_loader.dataset)})，已调整为{actual_batch_size}")
        # 如果需要，重新创建数据加载器
        if len(train_loader) == 0:
            train_loader, test_loader = prepare_data(args.dataset, actual_batch_size, use_cuda=(args.device=='cuda'))
    
    # 创建模型
    if args.model == 'mlp':
        model = SpikingMLP(
            input_size=input_size,
            hidden_size=args.hidden_size,
            output_size=output_size,
            n_timesteps=args.timesteps,
            neuron_type=args.neuron
        )
    elif args.model == 'cnn':
        # 对于CNN，假设输入是图像数据
        if args.dataset.lower() in ['cifar10', 'cifar100']:
            input_channels = 1  # 灰度图像
        elif args.dataset.lower() in ['mnist', 'fmnist', 'kmnist', 'emnist']:
            input_channels = 1
        else:
            input_channels = 1  # 默认为1通道
            
        model = SpikingCNN(
            input_channels=input_channels,
            input_size=input_size,
            output_size=output_size,
            n_timesteps=args.timesteps,
            neuron_type=args.neuron
        )
    
    print(f"创建{args.model.upper()}模型: {args.neuron}神经元, {args.timesteps}时间步长")
    
    # 创建保存路径
    model_save_path = f"{args.save_dir}/{args.dataset}_{args.model}_{args.neuron}_T{args.timesteps}.pth"
    fig_save_path = f"{args.save_dir}/{args.dataset}_{args.model}_{args.neuron}_T{args.timesteps}.png"
    
    # 训练模型
    model, train_losses, test_accuracies = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        save_path=model_save_path
    )
    
    # 绘制结果
    plot_results(train_losses, test_accuracies, fig_save_path)
    
    # 打印最佳性能
    best_acc = max(test_accuracies)
    best_epoch = test_accuracies.index(best_acc) + 1
    print(f"\n最佳测试准确率: {best_acc:.2f}% (Epoch {best_epoch})")

if __name__ == "__main__":
    main() 