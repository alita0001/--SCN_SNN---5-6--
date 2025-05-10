#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
伪逆计算方法比较工具
用于测试和比较不同生物启发的伪逆计算方法
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import linalg as LA
import sys

# 设置随机种子，确保每次运行结果一致
np.random.seed(42)  # 使用与其他文件相同的随机种子值

# from SCN_SNN import (
#     pseudo_inv,  # 原始伪逆方法
#     # homeostatic_pseudo_inverse,  # 基于突触稳态的伪逆
#     # bcm_closed_form,  # BCM理论的闭式解
#     # synaptic_tagging_capture,  # 突触标签捕获模型
#     # balanced_excitation_inhibition,  # 平衡抑制兴奋的解析解
#     # # 以下是需要迭代的方法，仅作参考
#     # stdp_weight_update,  # STDP启发的权重更新
#     # local_hebbian_learning,  # 局部霍姆学习
#     # dopamine_modulated_learning,  # 多巴胺调制学习
# )

import os
from matplotlib.font_manager import FontProperties

# 解决中文显示问题的代码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 检查字体文件是否存在
font_path = "SimHei.ttf"  # 默认字体路径
if not os.path.exists(font_path):
    # 如果当前目录没有字体文件，尝试使用系统默认字体
    try:
        # 尝试加载系统内的中文字体
        font = FontProperties(family='SimHei')
    except:
        print("警告: 找不到中文字体文件，图形中的中文可能无法正确显示")
        font = None
else:
    font = FontProperties(fname=font_path)


def pseudo_inv(A, reg):  # A拓展输入矩阵/隐层输出矩阵，reg正则化参数
    A_p = np.mat(A.T.dot(A) + reg * np.eye(A.shape[1])).I.dot(A.T)  # A.I求非奇异矩阵的逆，奇异时补上无穷小项
    # A_p = np.linalg.pinv(A.T.dot(A)).dot(A.T)
    return np.array(A_p)


########################################################################

# 以下是用户提供的额外伪逆计算方法

def hebbian_learning_pinv(A, target, lr=0.01, epochs=100):
    """基于赫布学习的伪逆计算方法

    Args:
        A: 隐层输出矩阵
        target: 目标输出
        lr: 学习率
        epochs: 训练轮数

    Returns:
        OutputWeight: 计算出的权重
    """
    W = np.random.randn(A.shape[1], target.shape[1]) * 0.01
    for _ in range(epochs):
        W += lr * A.T @ (target - A @ W)
    return W


def stdp_learning_alt(A, target, lr=0.01, epochs=100):
    """基于STDP的替代伪逆计算方法

    Args:
        A: 隐层输出矩阵
        target: 目标输出
        lr: 学习率
        epochs: 训练轮数

    Returns:
        OutputWeight: 计算出的权重
    """
    W = np.random.randn(A.shape[1], target.shape[1]) * 0.01
    for _ in range(epochs):
        dw = lr * (A.T @ target - A.T @ A @ W)
        W += dw
    return W


def lcl_learning_pinv(A, target, lr=0.01, epochs=100, sparsity=0.1):
    """基于局部竞争学习（LCL）的伪逆计算方法

    Args:
        A: 隐层输出矩阵
        target: 目标输出
        lr: 学习率
        epochs: 训练轮数
        sparsity: 稀疏度，控制突触权重的稀疏性

    Returns:
        OutputWeight: 计算出的权重
    """
    W = np.random.randn(A.shape[1], target.shape[1]) * 0.01
    for _ in range(epochs):
        W += lr * (A.T @ target - A.T @ A @ W)
        # 应用稀疏约束，保留绝对值大于阈值的权重
        W *= (np.abs(W) > sparsity)
    return W


def neural_dynamics_alt(A, target, alpha=0.01, iterations=1000):
    """基于神经动力学的替代伪逆计算方法

    Args:
        A: 隐层输出矩阵
        target: 目标输出
        alpha: 学习速率
        iterations: 迭代次数

    Returns:
        OutputWeight: 计算出的权重
    """
    W = np.random.randn(A.shape[1], target.shape[1]) * 0.01
    for _ in range(iterations):
        # 梯度下降优化最小二乘误差
        W += alpha * (A.T @ A @ W - A.T @ target)
    return W


def least_squares_direct(A, target, reg=1e-4):
    """直接最小二乘法计算伪逆

    Args:
        A: 隐层输出矩阵
        target: 目标输出
        reg: 正则化参数

    Returns:
        OutputWeight: 计算出的权重
    """
    try:
        # 直接计算正规方程解
        W = np.linalg.inv(A.T @ A + reg * np.eye(A.shape[1])) @ A.T @ target
        return W
    except:
        # 如果矩阵不可逆，使用SVD伪逆
        return np.linalg.pinv(A) @ target


########################################################################

def stdp_weight_update(A, target, learning_rate=0.01, epochs=100):
    """使用STDP启发的权重更新方法

    Args:
        A: 隐层输出矩阵
        target: 目标输出
        learning_rate: 学习率
        epochs: 训练轮数

    Returns:
        OutputWeight: 更新后的权重
    """
    # 初始化权重（可以使用小的随机值）
    OutputWeight = np.random.randn(A.shape[1], target.shape[1]) * 0.01

    for epoch in range(epochs):
        # 计算当前输出
        output = np.dot(A, OutputWeight)

        # 计算误差
        error = target - output

        # 计算权重更新（类似于STDP的机制）
        # 如果隐层神经元活动与目标输出正相关，增强连接
        # 如果隐层神经元活动与目标输出负相关，减弱连接
        delta_w = learning_rate * np.dot(A.T, error)

        # 更新权重
        OutputWeight += delta_w

        # 可以增加早停机制
        if np.mean(np.abs(error)) < 1e-4:
            break

    return OutputWeight


def local_hebbian_learning(A, target, learning_rate=0.01, epochs=100, decay=0.01):
    """使用局部霍姆学习规则更新权重

    Args:
        A: 隐层输出矩阵
        target: 目标输出
        learning_rate: 学习率
        epochs: 训练轮数
        decay: 权重衰减率（防止权重无限增长）

    Returns:
        OutputWeight: 更新后的权重
    """
    # 初始化权重
    OutputWeight = np.random.randn(A.shape[1], target.shape[1]) * 0.01

    for epoch in range(epochs):
        # 计算当前输出
        output = np.dot(A, OutputWeight)

        # 计算误差用于监控
        error = target - output

        # 霍姆学习更新（神经元同时激活时增强连接）
        # 引入误差指导，使霍姆学习可以最小化残差
        for i in range(A.shape[0]):
            # 计算神经元活动相关性，与误差加权
            delta_w = learning_rate * np.outer(A[i, :], error[i, :])

            # 权重更新
            OutputWeight += delta_w

        # 权重衰减（类似于突触稳定机制）
        OutputWeight *= (1 - decay)

        # 监控误差
        if epoch % 10 == 0:
            mse = np.mean(np.square(error))
            if mse < 1e-4:
                break

    return OutputWeight


def dopamine_modulated_learning(A, target, learning_rate=0.01, epochs=200):
    """使用基于奖励信号的多巴胺调制学习更新权重

    Args:
        A: 隐层输出矩阵
        target: 目标输出
        learning_rate: 学习率
        epochs: 训练轮数

    Returns:
        OutputWeight: 更新后的权重
    """
    # 初始化权重
    OutputWeight = np.random.randn(A.shape[1], target.shape[1]) * 0.01

    # 初始化预期奖励（负的误差平方和）
    expected_reward = -np.inf

    for epoch in range(epochs):
        # 计算当前输出
        output = np.dot(A, OutputWeight)

        # 计算误差
        error = target - output
        mse = np.mean(np.square(error))

        # 计算当前奖励（负的误差平方和，误差越小奖励越大）
        current_reward = -mse

        # 计算奖励预测误差（RPE）- 模拟多巴胺信号
        reward_prediction_error = current_reward - expected_reward

        # 多巴胺调制的学习率
        modulated_lr = learning_rate * (1.0 + np.tanh(reward_prediction_error))

        # 更新权重（根据奖励预测误差调整学习强度）
        delta_w = modulated_lr * np.dot(A.T, error)
        OutputWeight += delta_w

        # 更新预期奖励
        expected_reward = 0.9 * expected_reward + 0.1 * current_reward

        # 早停条件
        if mse < 1e-4:
            break

    return OutputWeight


def homeostatic_pseudo_inverse(A, target, reg_factor=0.01):
    """基于突触稳态的伪逆计算

    Args:
        A: 隐层输出矩阵
        target: 目标输出
        reg_factor: 正则化因子，类似于稳态调节强度

    Returns:
        OutputWeight: 计算出的权重
    """
    # 计算协方差矩阵（对应于突触活动相关性）
    covariance = A.T.dot(A)

    # 添加稳态调节项（类似于神经元自我调节机制）
    # 神经生物学中，突触强度会受到突触前、后神经元活动水平的调节
    activity_level = np.mean(A, axis=0).reshape(-1, 1)
    homeostasis_factor = np.exp(-activity_level) * reg_factor

    # 应用稳态调节到协方差矩阵的对角线上
    regularized_cov = covariance + np.diag(homeostasis_factor.flatten())

    # 计算伪逆并获取权重
    try:
        inv_cov = np.linalg.inv(regularized_cov)
    except:
        # 如果矩阵接近奇异，使用更强的正则化
        inv_cov = np.linalg.inv(regularized_cov + reg_factor * np.eye(A.shape[1]))

    # 计算输出权重
    OutputWeight = inv_cov.dot(A.T).dot(target)

    return OutputWeight


def bcm_closed_form(A, target, theta=None):
    """基于BCM理论的闭式解

    Args:
        A: 隐层输出矩阵
        target: 目标输出
        theta: 可塑性阈值，如果为None则自动计算

    Returns:
        OutputWeight: 计算出的权重
    """
    # 如果没有提供阈值，则以平均活动水平的平方作为阈值
    if theta is None:
        theta = np.mean(A ** 2, axis=0)

    # 计算BCM调节因子（post-synaptic活动减去阈值）
    bcm_factors = np.mean(A, axis=0) - theta

    # 应用BCM调节到相关性矩阵
    correlation = A.T.dot(target)
    scaled_correlation = correlation * (1 + np.tanh(bcm_factors.reshape(-1, 1)))

    # 计算归一化因子
    norm_factor = np.sum(A ** 2, axis=0).reshape(-1, 1)

    # 计算权重（类似于相关性除以自相关）
    OutputWeight = scaled_correlation / (norm_factor + 1e-6)

    return OutputWeight


def synaptic_tagging_capture(A, target, tag_threshold=0.5):
    """基于突触标签捕获模型的直接求解

    Args:
        A: 隐层输出矩阵
        target: 目标输出
        tag_threshold: 标签阈值，决定哪些突触会被加强

    Returns:
        OutputWeight: 计算出的权重
    """
    # 计算隐层与目标层的相关性（对应于潜在的突触变化）
    correlation = A.T.dot(target)

    # 归一化相关性
    norm_corr = correlation / (np.sqrt(np.sum(A ** 2, axis=0)).reshape(-1, 1) + 1e-6)

    # 生成突触标签（模拟只有强相关才会形成标签）
    synaptic_tags = np.abs(norm_corr) > tag_threshold

    # 计算权重捕获（只有被标记的突触才会捕获增强信号）
    capture_signal = np.sign(correlation) * np.mean(np.abs(correlation))

    # 最终权重是标签和捕获信号的组合
    OutputWeight = correlation * synaptic_tags + capture_signal * (1 - synaptic_tags) * 0.1

    # 归一化权重（模拟突触稳态）
    OutputWeight = OutputWeight / (np.sqrt(np.sum(OutputWeight ** 2, axis=1)).reshape(-1, 1) + 1e-6)

    return OutputWeight


def balanced_excitation_inhibition(A, target, balance_factor=0.5):
    """基于平衡抑制兴奋的解析解

    Args:
        A: 隐层输出矩阵
        target: 目标输出
        balance_factor: 平衡因子，控制兴奋抑制比例

    Returns:
        OutputWeight: 计算出的权重
    """
    # 计算输入与目标的协方差
    cov = A.T.dot(target)

    # 计算平均活动水平
    mean_activity = np.mean(A, axis=0).reshape(-1, 1)

    # 生成抑制和兴奋权重
    excitatory = np.maximum(cov, 0)  # 兴奋性权重（正值）
    inhibitory = np.minimum(cov, 0)  # 抑制性权重（负值）

    # 平衡兴奋抑制（使用平衡因子调节）
    exc_inh_ratio = balance_factor / (1 - balance_factor + 1e-6)
    balanced_inhibition = inhibitory * exc_inh_ratio

    # 最终权重考虑兴奋抑制平衡和平均活动水平
    OutputWeight = (excitatory + balanced_inhibition) / (mean_activity + 1e-6)

    return OutputWeight


# 添加神经动力学伪逆计算
def neural_dynamics_pinv(A, target, learning_rate=0.1, max_iterations=1000, convergence_threshold=1e-6):
    """
    基于神经动力学的伪逆计算
    参考: Zeng & Wang (1991), "A Dynamic Neural Network Method for the Computation of Matrix Pseudo-Inverse"

    Args:
        A: 隐层输出矩阵
        target: 目标输出
        learning_rate: 学习率
        max_iterations: 最大迭代次数
        convergence_threshold: 收敛阈值

    Returns:
        OutputWeight: 计算出的权重
    """
    # 初始化权重矩阵为A的转置（生物神经网络中的海博连接初始化）
    W = np.zeros((A.shape[1], A.shape[0]))

    # 动态神经网络迭代
    for i in range(max_iterations):
        # 计算残差
        residual = A.dot(W) - np.eye(A.shape[0])

        # 更新权重矩阵
        delta_W = -learning_rate * W.dot(residual)
        W += delta_W

        # 检查收敛
        if np.linalg.norm(delta_W) < convergence_threshold:
            break

    # 计算伪逆
    return W.dot(target)


# 添加脊髓中枢模式发生器启发的伪逆计算
def cpg_inspired_pinv(A, target, oscillation_periods=5, damping=0.9):
    """
    基于中枢模式发生器(CPG)的伪逆计算

    Args:
        A: 隐层输出矩阵
        target: 目标输出
        oscillation_periods: 振荡周期数
        damping: 阻尼系数

    Returns:
        OutputWeight: 计算出的权重
    """
    m, n = A.shape

    # 计算SVD分解，类似于神经系统中的特征提取
    try:
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
    except:
        # 如果SVD失败，添加微小扰动
        U, S, Vt = np.linalg.svd(A + 1e-10 * np.random.randn(*A.shape), full_matrices=False)

    # 应用CPG的振荡特性，通过振荡函数处理奇异值
    S_inv = np.zeros(S.shape)
    for i, s in enumerate(S):
        if s > 1e-10:
            # 使用带阻尼的振荡函数处理奇异值（模拟CPG振荡特性）
            oscillation = 0
            for p in range(oscillation_periods):
                phase = 2 * np.pi * p / oscillation_periods
                oscillation += np.cos(phase) * (damping ** p)

            S_inv[i] = (1.0 / s) * (1 + 0.01 * oscillation)

    # 构建伪逆
    A_pinv = Vt.T.dot(np.diag(S_inv)).dot(U.T)

    # 计算权重
    return A_pinv.dot(target)


# 添加预测编码伪逆计算
def predictive_coding_pinv(A, target, precision=0.1, iterations=100):
    """
    基于预测编码理论的伪逆计算
    参考: Rao & Ballard (1999), "Predictive coding in the visual cortex"

    Args:
        A: 隐层输出矩阵
        target: 目标输出
        precision: 预测精度
        iterations: 迭代次数

    Returns:
        OutputWeight: 计算出的权重
    """
    # 初始化权重
    W = np.zeros((A.shape[1], target.shape[1]))

    # 初始化预测误差
    prediction_error = target.copy()

    # 预测编码迭代
    for i in range(iterations):
        # 计算当前预测
        prediction = A.dot(W)

        # 更新预测误差
        prediction_error = target - prediction

        # 基于误差更新权重（类似于大脑中的预测编码机制）
        delta_W = precision * A.T.dot(prediction_error)
        W += delta_W

        # 自适应精度调整（类似于神经元注意力机制）
        precision = precision * (1 + 0.01 * np.tanh(np.mean(np.abs(prediction_error))))

    return W


# 添加突触元整合计算
def synaptic_meta_integration(A, target, reg_factor=0.01):
    """
    基于突触元整合的伪逆计算

    Args:
        A: 隐层输出矩阵
        target: 目标输出
        reg_factor: 正则化因子

    Returns:
        OutputWeight: 计算出的权重
    """
    # 计算不同类型的突触整合模式

    # 1. 长期增强 (LTP) 机制 - 相关性强化
    correlation = A.T.dot(target)

    # 2. 突触标定 (Synaptic Scaling) - 归一化
    synapse_scaling = np.sqrt(np.sum(A ** 2, axis=0)).reshape(-1, 1) + 1e-6

    # 3. 内稳态突触调节 (Homeostatic Regulation)
    mean_activity = np.mean(A, axis=0).reshape(-1, 1)
    homeostatic_factor = 1.0 / (1.0 + np.exp(-5 * (mean_activity - 0.5)))  # sigmoid调节

    # 4. 侧向抑制 (Lateral Inhibition)
    A_cov = A.T.dot(A)
    off_diagonal = A_cov - np.diag(np.diag(A_cov))
    lateral_inhibition = 1.0 / (1.0 + reg_factor * np.sum(np.abs(off_diagonal), axis=1).reshape(-1, 1))

    # 整合多种突触机制
    W = correlation * lateral_inhibition * homeostatic_factor / synapse_scaling

    return W


# 添加海马体自联想记忆模型
def hippocampal_auto_associative_pinv(A, target, sparsity=0.2):
    """
    基于海马体自联想记忆的伪逆计算

    Args:
        A: 隐层输出矩阵
        target: 目标输出
        sparsity: 稀疏度参数，控制激活神经元的比例

    Returns:
        OutputWeight: 计算出的权重
    """
    # 稀疏化隐层表示（模拟海马体中的稀疏编码）
    A_sparse = A.copy()
    for i in range(A.shape[0]):
        # 保留每行最大的sparsity%的元素
        threshold = np.percentile(A[i, :], 100 * (1 - sparsity))
        A_sparse[i, A[i, :] < threshold] = 0

    # 计算海马体CA3区域样式的自联想矩阵
    # 在生物学中，这种计算可能通过Schaffer侧枝路径实现
    A_auto = A_sparse.T.dot(A_sparse)

    # 添加正则化（类似于牙状回区域的抑制性调节）
    A_auto_reg = A_auto + np.eye(A_auto.shape[0]) * (np.trace(A_auto) / A_auto.shape[0]) * 0.1

    try:
        # 计算逆（类似于CA3-CA1路径的信息处理）
        A_inv = np.linalg.inv(A_auto_reg)
    except:
        # 如果计算逆失败，使用伪逆
        A_inv = np.linalg.pinv(A_auto_reg)

    # 最终权重计算
    return A_inv.dot(A_sparse.T).dot(target)


def RMSE(output, target):
    """计算均方根误差"""
    E = output - target
    N = E.shape[0]
    return np.sqrt(np.sum(np.sum(E ** 2, axis=0) / N))


def generate_test_data(num_samples=800, seed=42):
    """生成测试数据集"""
    np.random.seed(seed)

    # 简单函数
    x_simple = np.linspace(-1, 1, num_samples).reshape(-1, 1)
    y_simple = x_simple ** 3

    # 复杂函数（多个高斯组合）
    x_complex = np.linspace(0, 1, num_samples).reshape(-1, 1)
    term1 = 0.2 * np.exp(-np.square(10 * x_complex - 4))
    term2 = 0.5 * np.exp(-np.square(80 * x_complex - 40))
    term3 = 0.3 * np.exp(-np.square(80 * x_complex - 20))
    y_complex = term1 + term2 + term3

    # 带噪声的数据
    x_noisy = np.linspace(-1, 1, num_samples).reshape(-1, 1)
    y_noisy = np.sin(2 * np.pi * x_noisy) + 0.2 * np.random.randn(num_samples, 1)

    # 高维数据
    x_high_dim = np.random.rand(num_samples, 20)
    w_true = np.random.randn(20, 1)
    y_high_dim = x_high_dim.dot(w_true) + 0.1 * np.random.randn(num_samples, 1)

    return {
        "simple": (x_simple, y_simple),
        "complex": (x_complex, y_complex),
        "noisy": (x_noisy, y_noisy),
        "high_dim": (x_high_dim, y_high_dim)
    }


def test_method(method_func, input_data, target_data, method_name, reg_factor=1e-6):
    """测试指定的伪逆计算方法"""
    start_time = time.time()

    try:
        if method_name == "原始伪逆方法":
            # 原始方法需要特殊处理
            A_p = pseudo_inv(input_data, reg_factor)
            weights = np.dot(A_p, target_data)
        else:
            # 其他方法直接调用
            weights = method_func(input_data, target_data)

        output = np.dot(input_data, weights)
        rmse = RMSE(output, target_data)
        compute_time = time.time() - start_time

        return {
            "weights": weights,
            "output": output,
            "rmse": rmse,
            "time": compute_time,
            "success": True
        }
    except Exception as e:
        return {
            "weights": None,
            "output": None,
            "rmse": float('inf'),
            "time": time.time() - start_time,
            "success": False,
            "error": str(e)
        }


def run_all_tests():
    """运行所有测试并生成比较报告"""
    # 定义所有要测试的方法
    methods = {
        "原始伪逆方法": pseudo_inv,
        "突触稳态伪逆": homeostatic_pseudo_inverse,
        "BCM理论闭式解": bcm_closed_form,
        "突触标签捕获模型": synaptic_tagging_capture,
        "平衡抑制兴奋解析解": balanced_excitation_inhibition,
        "神经动力学伪逆": neural_dynamics_pinv,
        "中枢模式发生器伪逆": cpg_inspired_pinv,
        "预测编码伪逆": predictive_coding_pinv,
        "突触元整合伪逆": synaptic_meta_integration,
        "海马体自联想记忆伪逆": hippocampal_auto_associative_pinv,
        "赫布学习伪逆": hebbian_learning_pinv,
        "STDP替代伪逆": stdp_learning_alt,
        "局部竞争学习伪逆": lcl_learning_pinv,
        "神经动力学替代伪逆": neural_dynamics_alt,
        "直接最小二乘法": least_squares_direct,
        "Hebb学习伪逆": hebbian_pseudo_inv,
        "STDP伪逆": stdp_pseudo_inv,
        "递归伪逆": recurrent_pseudo_inv,
    }

    # 生成测试数据
    test_data = generate_test_data()

    # 用于存储所有结果的字典
    results = {}

    # 对每种数据类型进行测试
    for data_name, (X, y) in test_data.items():
        print(f"测试数据集: {data_name}")
        results[data_name] = {}

        # 使用每种方法测试
        for method_name, method_func in methods.items():
            print(f"  测试方法: {method_name}")
            result = test_method(method_func, X, y, method_name)
            results[data_name][method_name] = result

            if result["success"]:
                print(f"    RMSE: {result['rmse']:.6f}, 计算时间: {result['time']:.6f}秒")
            else:
                print(f"    失败: {result['error']}")

        print("\n" + "-" * 60 + "\n")

    return results


def plot_results(results):
    """绘制结果比较图表"""
    data_types = list(results.keys())
    method_names = list(results[data_types[0]].keys())

    # 准备RMSE数据
    rmse_data = {}
    time_data = {}

    for data_name in data_types:
        rmse_data[data_name] = []
        time_data[data_name] = []

        for method_name in method_names:
            result = results[data_name][method_name]
            if result["success"]:
                rmse_data[data_name].append(result["rmse"])
                time_data[data_name].append(result["time"])
            else:
                rmse_data[data_name].append(float('nan'))
                time_data[data_name].append(float('nan'))

    # 绘制RMSE比较图
    plt.figure(figsize=(15, 10))

    for i, data_name in enumerate(data_types):
        plt.subplot(2, 2, i + 1)

        x = np.arange(len(method_names))
        plt.bar(x, rmse_data[data_name])
        plt.xticks(x, method_names, rotation=45, ha='right')
        plt.title(f'{data_name} 数据集的RMSE比较')
        plt.ylabel('RMSE')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # 调整布局
        plt.tight_layout()

    plt.savefig("pinv_methods_rmse_comparison.png")
    plt.close()

    # 绘制计算时间比较图
    plt.figure(figsize=(15, 10))

    for i, data_name in enumerate(data_types):
        plt.subplot(2, 2, i + 1)

        x = np.arange(len(method_names))
        plt.bar(x, time_data[data_name])
        plt.xticks(x, method_names, rotation=45, ha='right')
        plt.title(f'{data_name} 数据集的计算时间比较')
        plt.ylabel('时间 (秒)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # 调整布局
        plt.tight_layout()

    plt.savefig("pinv_methods_time_comparison.png")
    plt.close()

    # 绘制拟合曲线对比（仅对简单数据集）
    if "complex" in results:
        X, y = generate_test_data()["complex"]

        plt.figure(figsize=(15, 6))
        plt.subplot(1, 1, 1)

        # 绘制真实值
        plt.plot(X, y, 'k-', linewidth=2, label='真实值')

        # 最多绘制5种方法的拟合结果
        colors = ['r', 'g', 'b', 'c', 'm']
        shown_methods = 0

        for i, method_name in enumerate(method_names):
            result = results["complex"][method_name]
            if result["success"] and shown_methods < 5:
                plt.plot(X, result["output"], f'{colors[shown_methods]}--', linewidth=1.5, label=f'{method_name}')
                shown_methods += 1

        plt.title('不同方法在简单数据集上的拟合曲线对比')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig("pinv_methods_fitting_comparison.png")
        plt.close()

    print("结果图表已保存为PNG文件。")


def main():
    print("=" * 80)
    print("伪逆计算方法比较工具".center(70))
    print("=" * 80)

    print("开始测试所有伪逆方法...\n")
    results = run_all_tests()

    print("生成比较图表...\n")
    plot_results(results)

    print("测试完成！")


if __name__ == "__main__":
    main()


# 添加基于Hebb学习的伪逆计算
def hebbian_pseudo_inv(A, target, reg=0.001, lr=0.01, epochs=500, decay=0.99):
    """
    基于Hebb学习规则的伪逆计算方法

    Args:
        A: 隐层输出矩阵
        target: 目标输出
        reg: 正则化参数
        lr: 学习率
        epochs: 训练轮数

    Returns:
        OutputWeight: 计算出的权重
    """
    #W = np.zeros((A.shape[1], A.shape[1]))  # 权重矩阵
    # for _ in range(epochs):
    #     W += lr * (A.T.dot(A)  - decay * W) # Hebb学习更新
    W = A.T.dot(A)
    W += reg * np.eye(A.shape[1])  # 添加正则化
    A_p = np.linalg.inv(W).dot(A.T)  # 计算伪逆
    return np.dot(A_p, target)



# 添加基于STDP的伪逆计算
def stdp_pseudo_inv(A, target, reg=0.001, lr=0.01, decay=0.99, epochs=500):
    """
    基于突触时间依赖可塑性(STDP)的伪逆计算方法

    Args:
        A: 隐层输出矩阵
        target: 目标输出
        reg: 正则化参数
        lr: 学习率
        decay: 权重衰减系数
        epochs: 训练轮数

    Returns:
        OutputWeight: 计算出的权重
    """
    W = np.zeros((A.shape[1], A.shape[1]))  # 权重初始化
    for _ in range(epochs):
        W += lr * (A.T @ A - decay * W)  # STDP更新
    W += reg * np.eye(A.shape[1])  # 正则化
    A_p = np.linalg.inv(W) @ A.T  # 计算伪逆
    return np.dot(A_p, target)


# 添加基于递归网络的伪逆计算
def recurrent_pseudo_inv(A, target, reg=0.001, lr=0.01, epochs=500):
    """
    基于递归神经网络的伪逆计算方法

    Args:
        A: 隐层输出矩阵
        target: 目标输出
        reg: 正则化参数
        lr: 学习率
        epochs: 训练轮数

    Returns:
        OutputWeight: 计算出的权重
    """
    W = np.zeros((A.shape[1], A.shape[1]))
    for _ in range(epochs):
        W += lr * (A.T @ A - W)  # 递归更新
    W += reg * np.eye(A.shape[1])  # 正则化
    A_p = np.linalg.inv(W) @ A.T  # 计算伪逆
    return np.dot(A_p, target)
