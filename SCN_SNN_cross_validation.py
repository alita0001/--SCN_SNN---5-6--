# -*- coding: utf-8 -*-
import numpy as np
from math import sqrt
from sklearn import preprocessing
from scipy import linalg as LA
import time
import matplotlib.pyplot as plt
import image_process_copy

# 设置随机种子，确保每次运行结果一致
np.random.seed(6)  # 42是一个常用的随机种子值，可以更改为任意整数

from pinv_methods_comparison import (
    homeostatic_pseudo_inverse,  # 基于突触稳态的伪逆
    bcm_closed_form,  # BCM理论的闭式解
    synaptic_tagging_capture,  # 突触标签捕获模型
    balanced_excitation_inhibition,  # 平衡抑制兴奋的解析解
    # 以下是需要迭代的方法，仅作参考
    stdp_weight_update,  # STDP启发的权重更新
    local_hebbian_learning,  # 局部霍姆学习
    dopamine_modulated_learning,  # 多巴胺调制学习
)

####################### 新增的SNN模块 ########################
class LIFNeuron:
    def __init__(self, threshold=1.0, tau=10.0, dt=0.1):
        self.threshold = threshold  # 脉冲发放阈值
        self.tau = tau  # 膜时间常数 这里看成膜时间常数的倒数
        self.dt = dt  # 时间步长
        self.membrane_potential = 0  # 膜电位
        self.H = 0 # 膜电位中间状态

    def reset(self):
        self.membrane_potential = 0

    def forward(self, I):
        # 更新膜电位: tau * dV/dt = -V + I
        # self.H = (1-self.tau)*self.membrane_potential + self.tau*I
        self.membrane_potential += (-self.membrane_potential + I) / self.tau * self.dt

        # 发放脉冲并重置
        if self.membrane_potential > self.threshold:
            spike = 1
            self.membrane_potential = 0  # 硬重置
        else:
            spike = 0
        return spike


def normalize_data(data):
    # 归一化数据到0-1范围
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)#*0.99 + 0.01
    return normalized_data

def poisson_encoding(data, timesteps):
    # 归一化数据
    normalized_data = normalize_data(data)

    # 将数据编码为泊松脉冲序列 [batchsize, features] -> [timesteps, batchsize, features]
    poisson_pulses = np.random.rand(timesteps, data.shape[0], data.shape[1]) < normalized_data[np.newaxis, :, :]

    return poisson_pulses


def lif_layer(input_spikes, weights, bias, time_steps=10, threshold=1.0, tau=50.0, dt=1.0):
    """
    向量化实现的LIF层
    Args:
        input_spikes: [time_steps, batch_size, input_dim]
        weights: [output_dim, input_dim]
        bias: [output_dim, 1]
    Returns:
        firing_rate: [batch_size, output_dim]
    """
    # 预计算所有时间步的输入电流 [time_steps, batch_size, output_dim]
    I = np.einsum('tbi,oi->tbo', input_spikes, weights) + bias.T  # 向量化矩阵乘法


    # 初始化膜电位和脉冲记录
    batch_size, output_dim = I.shape[1], weights.shape[0]
    membrane = np.zeros((batch_size, output_dim))
    spikes = np.zeros((time_steps, batch_size, output_dim))

    for t in range(time_steps):
        # 更新膜电位
        membrane += (-membrane + I[t]) / tau * dt

        # 检测脉冲并硬重置
        spike = (membrane > threshold).astype(float)
        spikes[t] = spike
        membrane = membrane * (1 - spike)  # 发放后重置为0 硬重置机制
        # membrane = membrane - spike * threshold  # 减去阈值代替归零 软重置机制

    return np.mean(spikes, axis=0)

# def lif_layer(input_spikes, weights, bias, time_steps=10, threshold=1.0, tau=10.0, dt=1.0):
#     """
#     向量化实现的LIF层
#     Args:
#         input_spikes: [time_steps, batch_size, input_dim]
#         weights: [output_dim, input_dim]
#         bias: [output_dim, 1]
#     Returns:
#         firing_rate: [batch_size, output_dim]
#     """
#     # 预计算所有时间步的输入电流 [time_steps, batch_size, output_dim]
#     I = np.einsum('tbi,oi->tbo', input_spikes, weights) + bias.T  # 向量化矩阵乘法
#
#
#     # 初始化膜电位和脉冲记录
#     batch_size, output_dim = I.shape[1], weights.shape[0]
#     membrane = np.zeros((batch_size, output_dim))
#     spikes = np.zeros((time_steps, batch_size, output_dim))
#
#     for t in range(time_steps):
#         # 更新膜电位
#         membrane += (-membrane + I[t]) / tau * dt
#
#         # 检测脉冲并硬重置
#         # spike = (membrane > threshold).astype(float)
#         # spikes[t] = spike
#         # membrane = membrane * (1 - spike)  # 发放后重置为0 硬重置机制
#         # membrane = membrane - spike * threshold  # 减去阈值代替归零 软重置机制
#
#     return membrane


##############################################################

# 回归精度计算保持不变
def RMSE(train_output, train_y):
    E = train_output - train_y
    N = E.shape[0]
    Residual = sqrt(np.sum(np.sum(E ** 2, axis=0) / N))
    return Residual

# 分类精度
def classification_accuracy(predictLabel, Label):
    count = 0
    label = Label.argmax(axis=1)
    prediction = predictLabel.argmax(axis=1)
    for j in list(range(Label.shape[0])):
        if label[j] == prediction[j]:
            count += 1
    return round(count / len(Label), 5)

def pseudo_inv(A, reg):  # A拓展输入矩阵/隐层输出矩阵，reg正则化参数
    A_p = np.mat(A.T.dot(A) + reg * np.eye(A.shape[1])).I.dot(A.T)  # A.I求非奇异矩阵的逆，奇异时补上无穷小项
    # A_p = np.linalg.pinv(A.T.dot(A)).dot(A.T)
    return np.array(A_p)


def SCN_regression_SNN(train_x, train_y, test_x, test_y, dl, delta, c, Lambdas, r, verbose, Lmax, tol, Tmax,
                       time_steps=5000):
    # 脉冲编码输入数据
    encoding_start = time.time()  # 计时开始：脉冲编码阶段
    train_x_spikes = poisson_encoding(train_x, time_steps)
    test_x_spikes = poisson_encoding(test_x, time_steps)
    encoding_time = time.time() - encoding_start
    print(f'数据脉冲编码阶段耗时: {encoding_time:.4f}s')

    L = 0
    k = 0
    eL_q = train_y
    Error = RMSE(eL_q, np.zeros([eL_q.shape[0], 1]))
    ErrorList = []
    W = np.empty((0, train_x.shape[1]))
    B = np.empty((0, 1))
    HL = np.empty((train_x.shape[0], 0))
    Omega = np.empty((0, 1))
    W_opt = np.empty((0, train_x.shape[1]))
    B_opt = np.empty((0, 1))

    # 记录各阶段耗时统计
    total_hidden_layer_time = 0
    total_lif_layer_time = 0 
    total_weight_update_time = 0
    total_output_weight_time = 0

    time_start = time.time() # 计时开始
    while L < Lmax and Error > tol and r<1:
        iteration_start = time.time()  # 计时开始：每次迭代
        
        if L % verbose == 0:
            print('#L: {}\t RMSE: {:.4f} \r'.format(L, Error))

        hidden_layer_start = time.time()  # 计时开始：隐藏层更新阶段
        
        for Lambda in Lambdas:
            lambda_start = time.time()  # 计时开始：每个Lambda值
            
            for t in range(0, Tmax):
                # InputWeight = Lambda * (2 * np.random.rand(delta, train_x.shape[1]))
                # InputBias = Lambda * (2 * np.random.rand(delta, 1))
                InputWeight = Lambda * (2 * np.random.rand(delta, train_x.shape[1]) - 1)
                InputBias = Lambda * (2 * np.random.rand(delta, 1) - 1)

                # 使用LIF层计算发放率作为隐层输出
                lif_start = time.time()  # 计时开始：LIF层计算
                H = lif_layer(train_x_spikes, InputWeight, InputBias, time_steps)
                lif_time = time.time() - lif_start
                total_lif_layer_time += lif_time
                
                print(f'发放率：{H[1]}')
                Mu = (1 - r) / (L + delta)
                if delta == 1:
                    ksi = ((eL_q.T.dot(H)) ** 2) / (H.T.dot(H)) - (1 - r) * (eL_q.T.dot(eL_q))
                else:
                    ksi = np.dot((eL_q.T.dot(H)), pseudo_inv(H.T.dot(H), c)).dot((H.T.dot(eL_q))) - (1 - r) * np.dot(eL_q.T, eL_q)

                if ksi > 0:
                    Omega = np.concatenate((Omega, ksi), axis=0)
                    W = np.concatenate((W, InputWeight), axis=0)
                    B = np.concatenate((B, InputBias), axis=0)
            
            lambda_time = time.time() - lambda_start
            print(f'Lambda={Lambda}耗时: {lambda_time:.4f}s')

            if len(Omega) >= 1:
                update_start = time.time()  # 计时开始：权重更新阶段
                index = np.argmax(Omega)
                W_opt_L = W[index * delta: (index + 1) * delta, :]
                B_opt_L = B[index * delta: (index + 1) * delta, :]
                H_opt = lif_layer(train_x_spikes, W_opt_L, B_opt_L, time_steps)
                W_opt = np.concatenate((W_opt, W_opt_L), axis=0)
                B_opt = np.concatenate((B_opt, B_opt_L), axis=0)
                HL = np.concatenate((HL, H_opt), axis=1)
                L += delta
                k += 1
                Omega = np.empty((0, 1))
                W = np.empty((0, train_x.shape[1]))
                B = np.empty((0, 1))
                update_time = time.time() - update_start
                total_weight_update_time += update_time
                print(f'权重更新阶段耗时: {update_time:.4f}s')
                break
            else:
                Tau = np.random.rand(1, 1) * (1 - r)
                r += Tau
                if r >= 1:
                    print("r已经为1，网络后续节点增加已经无法改进网络")
                    break
                print(f'当前的r:{r}')
        
        hidden_layer_time = time.time() - hidden_layer_start
        total_hidden_layer_time += hidden_layer_time
        print(f'隐藏层更新阶段总耗时: {hidden_layer_time:.4f}s')
        
        print('当前隐藏层神经元数量L: {}\r'.format(L))
        if dl == 1 and L == delta:
            A = np.concatenate((train_x, HL), axis=1)
            HL = A
        else:
            A = HL

        output_weight_start = time.time()  # 计时开始：输出权重计算阶段
        # 原始代码
        # A_p = pseudo_inv(A, c)
        # OutputWeight = np.dot(A_p, train_y)
        
        # 替换为生物启发算法，例如：
        #OutputWeight = dopamine_modulated_learning(A, train_y)
        # 或
        #OutputWeight = stdp_weight_update(A, train_y)
        # 或
        #OutputWeight = local_hebbian_learning(A, train_y)
        # 替换为生物启发直接求解算法，例如：
        OutputWeight = homeostatic_pseudo_inverse(A, train_y, reg_factor=c)
        # 或
        #OutputWeight = bcm_closed_form(A, train_y)
        # 或
        #OutputWeight = synaptic_tagging_capture(A, train_y)
        # 或
        #OutputWeight = balanced_excitation_inhibition(A, train_y)
        train_output = np.dot(A, OutputWeight)
        eL_q = train_output - train_y
        Error = RMSE(train_output, train_y)
        ErrorList = np.append(ErrorList, Error)
        output_weight_time = time.time() - output_weight_start
        total_output_weight_time += output_weight_time
        print(f'输出权重计算阶段耗时: {output_weight_time:.4f}s')
        
        iteration_time = time.time() - iteration_start
        print(f'本次迭代总耗时: {iteration_time:.4f}s\n{"-"*40}')

    print('After {} iterations, the number of hidden nodes is {},权重参数的shape为{}'.format(k, L, W_opt.shape))
    time_end = time.time()
    train_time = time_end - time_start
    train_acc = Error
    print('Training accurate is', train_acc)
    print('Training time is ', train_time, 's')
    
    # 输出各阶段耗时统计
    print('\n训练各阶段耗时统计:')
    print(f'数据脉冲编码阶段总耗时: {encoding_time:.4f}s')
    print(f'隐藏层更新阶段总耗时: {total_hidden_layer_time:.4f}s')
    print(f'LIF层计算总耗时: {total_lif_layer_time:.4f}s')
    print(f'权重更新阶段总耗时: {total_weight_update_time:.4f}s')
    print(f'输出权重计算阶段总耗时: {total_output_weight_time:.4f}s')
    print(f'平均每次迭代耗时: {train_time/k if k>0 else 0:.4f}s')

    # 测试过程（使用脉冲编码后的输入）
    time_start = time.time()  # 计时开始
    
    test_lif_start = time.time()  # 计时开始：测试LIF层计算
    tempH_test = lif_layer(test_x_spikes, W_opt, B_opt, time_steps)
    test_lif_time = time.time() - test_lif_start
    print(f'测试LIF层计算耗时: {test_lif_time:.4f}s')
    
    if dl == 1:
        A_test = np.hstack([test_x, tempH_test])
    else:
        A_test = tempH_test
    
    test_output_start = time.time()  # 计时开始：测试输出计算
    test_output = np.dot(A_test, OutputWeight)
    test_acc = RMSE(test_output, test_y)
    test_output_time = time.time() - test_output_start
    print(f'测试输出计算耗时: {test_output_time:.4f}s')
    
    time_end = time.time()
    test_time = time_end - time_start
    print('Testing accurate is', test_acc)
    print('Testing time is ', test_time, 's')

    plt.figure(1)
    plt.plot(test_x, test_output, 'r.-', label="Estimation")
    plt.plot(test_x, test_y, 'b.-', label="Real")
    plt.legend(loc='upper right')
    plt.xlabel('Inputs')
    plt.ylabel('Outputs')
    plt.title('SCN-SNN')

    plt.figure(2)
    plt.subplot(211)
    plt.plot(test_x, test_output)
    plt.subplot(212)
    plt.plot(test_x, test_y)
    plt.show()

    plt.figure(3)
    plt.plot( ErrorList, 'r.-', label="Estimation")
    plt.xlabel('The number of k')
    plt.ylabel('Residual error')
    plt.title('SCN—SNN')
    plt.show()

    return train_acc, train_time, test_acc, test_time
# 以下是SCN_classification_SNN函数中添加性能测试代码的修改

def SCN_classification_SNN(train_x, train_y, test_x, test_y, dl, delta, c, Lambdas, r, verbose, Lmax, tol, Tmax,
                       time_steps=200):
    # 脉冲编码输入数据
    encoding_start = time.time()  # 计时开始：脉冲编码阶段
    train_x_spikes = poisson_encoding(train_x, time_steps)
    test_x_spikes = poisson_encoding(test_x, time_steps)

    # # 对训练数据进行归一化
    # train_x = np.mean(train_x_spikes, axis=0)
    # test_x = np.mean(test_x_spikes, axis=0)

    encoding_time = time.time() - encoding_start
    print(f'数据脉冲编码阶段耗时: {encoding_time:.4f}s')

    L = 0
    k = 0
    eL_1 = train_y
    Error = classification_accuracy(eL_1, np.zeros((train_x.shape[0], train_y.shape[1])))  # 不妨与0比较，最终趋于1
    AccList = []
    W = np.empty((0, train_x.shape[1]))
    B = np.empty((0, 1))
    HL = np.empty((train_x.shape[0], 0))
    Omega = np.empty((0, 1))
    W_opt = np.empty((0, train_x.shape[1]))
    B_opt = np.empty((0, 1))

    # 记录各阶段耗时统计
    total_hidden_layer_time = 0
    total_lif_layer_time = 0 
    total_weight_update_time = 0
    total_output_weight_time = 0
    direct_link = 1
    # 用于记录每个节点数量对应的准确率
    node_counts = []
    train_accuracies = []
    test_accuracies = []

    time_start = time.time() # 计时开始
    while L < Lmax and Error > tol:
        iteration_start = time.time()  # 计时开始：每次迭代
        
        if L % verbose == 0:
            print('#k: {}\t #L: {}\t Acc: {:.4f} \r'.format(k, L, Error))

        hidden_layer_start = time.time()  # 计时开始：隐藏层更新阶段
        
        for Lambda in Lambdas:
            lambda_start = time.time()  # 计时开始：每个Lambda值
            
            print('Lambda: {}'.format(Lambda))
            for t in range(0, Tmax):
                print(t)
                # InputWeight = Lambda * (2 * np.random.rand(delta, train_x.shape[1]) - 1)
                # InputBias = 0.0 * (2 * np.random.rand(delta, 1) - 1)
                # 使用高斯分布进行参数初始化
                # InputWeight = np.random.randn(delta, train_x.shape[1]) * Lambda
                # InputBias = np.random.randn(delta, 1) * 0.0

                # 尝试使用何凯明初始化方法
                # 使用何凯明初始化方法 (He initialization)
                # 对于ReLU激活函数，标准差应为sqrt(2/n)，其中n是输入特征数量
                fan_in = train_x.shape[1]  # 输入特征数量
                std_dev = np.sqrt(2.0 / fan_in)
                InputWeight = np.random.randn(delta, train_x.shape[1]) * std_dev * Lambda
                # 偏置项通常初始化为0或很小的值
                InputBias = np.zeros((delta, 1))
                # print(f'使用何凯明初始化方法，标准差: {std_dev:.6f}')



                # 使用LIF层计算发放率作为隐层输出
                print(1)
                lif_start = time.time()  # 计时开始：LIF层计算
                H = lif_layer(train_x_spikes, InputWeight, InputBias, time_steps)
                lif_time = time.time() - lif_start
                total_lif_layer_time += lif_time
                print(f'LIF层计算耗时: {lif_time:.4f}s')
                
                print(f'发放率：{H[1]}')
                Mu = (1 - r) / (L + delta)

                # m分类对应m输出问题
                ksi_start = time.time()  # 计时开始：ksi计算
                ksi = np.zeros((1, train_y.shape[1]))
                for q in range(0, train_y.shape[1]):
                    eL_q = eL_1[:, q].reshape(-1, 1)
                    if delta == 1:
                        ksi = ((eL_q.T.dot(H)) ** 2) / (H.T.dot(H)+c) - (1 - r) * (eL_q.T.dot(eL_q))
                    else:
                        HTH = H.T.dot(H)
                        eLtH = eL_q.T.dot(H)
                        H_pseudo_inv = pseudo_inv(HTH, c)
                        term1 = np.dot(eLtH, H_pseudo_inv).dot((H.T.dot(eL_q)))
                        ksi = term1 - (1 - r) * np.dot(eL_q.T, eL_q)
                        # ksi = np.dot((eL_q.T.dot(H)), pseudo_inv(H.T.dot(H), c)).dot((H.T.dot(eL_q))) - (1 - r) * np.dot(eL_q.T, eL_q)
                ksi_time = time.time() - ksi_start
                print(f'ksi计算耗时: {ksi_time:.4f}s')

                if np.min(ksi) > 0:
                    ksi = np.sum(ksi).reshape(-1, 1)
                    Omega = np.concatenate((Omega, ksi), axis=0)
                    W = np.concatenate((W, InputWeight), axis=0)
                    B = np.concatenate((B, InputBias), axis=0)
                    break
            
            lambda_time = time.time() - lambda_start
            print(f'Lambda={Lambda}耗时: {lambda_time:.4f}s')

            if len(Omega) >= 1:
                update_start = time.time()  # 计时开始：权重更新阶段
                index = np.argmax(Omega)
                W_opt_L = W[index * delta: (index + 1) * delta, :]
                B_opt_L = B[index * delta: (index + 1) * delta, :]
                H_opt = lif_layer(train_x_spikes, W_opt_L, B_opt_L, time_steps)
                W_opt = np.concatenate((W_opt, W_opt_L), axis=0)
                B_opt = np.concatenate((B_opt, B_opt_L), axis=0)
                HL = np.concatenate((HL, H_opt), axis=1)
                L += delta
                k += 1
                Omega = np.empty((0, 1))
                W = np.empty((0, train_x.shape[1]))
                B = np.empty((0, 1))
                update_time = time.time() - update_start
                total_weight_update_time += update_time
                print(f'权重更新阶段耗时: {update_time:.4f}s')
                break
            else:
                Tau = np.random.rand(1, 1) * (1 - r)
                r += Tau
                print(f'当前的r:{r}')
        
        hidden_layer_time = time.time() - hidden_layer_start
        total_hidden_layer_time += hidden_layer_time
        print(f'隐藏层更新阶段总耗时: {hidden_layer_time:.4f}s')

        print('当前隐藏层神经元数量L: {}\r'.format(L))
        # if dl == 1 and L == delta:
        if dl == 1 and direct_link == 1:
            direct_link = 0
            A = np.concatenate((train_x, HL), axis=1)
            HL = A
        else:
            A = HL

        output_weight_start = time.time()  # 计时开始：输出权重计算阶段
        # 原始代码
        # A_p = pseudo_inv(A, c)
        # OutputWeight = np.dot(A_p, train_y)
        
        # 替换为生物启发算法
        OutputWeight = homeostatic_pseudo_inverse(A, train_y, reg_factor=c)
        
        train_output = np.dot(A, OutputWeight)
        eL_1 = train_output - train_y
        Error = classification_accuracy(train_output, train_y)
        AccList = np.append(AccList, Error)
        output_weight_time = time.time() - output_weight_start
        total_output_weight_time += output_weight_time
        print(f'输出权重计算阶段耗时: {output_weight_time:.4f}s')
        
        # 每增加节点后评估模型性能
        # 测试过程（使用脉冲编码后的输入）
        test_lif_start = time.time()
        tempH_test = lif_layer(test_x_spikes, W_opt, B_opt, time_steps)
        test_lif_time = time.time() - test_lif_start
        
        if dl == 1:
            A_test = np.hstack([test_x, tempH_test])
        else:
            A_test = tempH_test
        
        print(f'输出权重参数的shape为{OutputWeight.shape}')
        print(f'{test_x.shape = }')
        print(f'{tempH_test.shape = }')
        print(f'{A_test.shape = }')

        test_output = np.dot(A_test, OutputWeight)
        test_acc = classification_accuracy(test_output, test_y)
        
        # 记录当前节点数量和对应的准确率
        node_counts.append(L)
        train_accuracies.append(Error)
        test_accuracies.append(test_acc)
        
        print(f'节点数: {L}, 训练准确率: {Error:.4f}, 测试准确率: {test_acc:.4f}')
        
        iteration_time = time.time() - iteration_start
        print(f'本次迭代总耗时: {iteration_time:.4f}s\n{"-"*40}')

    print('After {} iterations, the number of hidden nodes is {},权重参数的shape为{}'.format(k, L, W.shape))
    time_end = time.time()
    train_time = time_end - time_start
    train_acc = Error
    print('Training accurate is {:.4f}'.format(train_acc * 100), '%')
    print('Training time is {:.4f}'.format(train_time), 's')
    
    # 输出各阶段耗时统计
    print('\n训练各阶段耗时统计:')
    print(f'数据脉冲编码阶段总耗时: {encoding_time:.4f}s')
    print(f'隐藏层更新阶段总耗时: {total_hidden_layer_time:.4f}s')
    print(f'LIF层计算总耗时: {total_lif_layer_time:.4f}s')
    print(f'权重更新阶段总耗时: {total_weight_update_time:.4f}s')
    print(f'输出权重计算阶段总耗时: {total_output_weight_time:.4f}s')
    print(f'平均每次迭代耗时: {train_time/k if k>0 else 0:.4f}s')

    # 测试过程（使用脉冲编码后的输入）
    time_start = time.time()  # 计时开始
    
    test_lif_start = time.time()  # 计时开始：测试LIF层计算
    tempH_test = lif_layer(test_x_spikes, W_opt, B_opt, time_steps)
    test_lif_time = time.time() - test_lif_start
    print(f'测试LIF层计算耗时: {test_lif_time:.4f}s')
    
    if dl == 1:
        A_test = np.hstack([test_x, tempH_test])
    else:
        A_test = tempH_test
    
    test_output_start = time.time()  # 计时开始：测试输出计算
    test_output = np.dot(A_test, OutputWeight)
    test_acc = classification_accuracy(test_output, test_y)
    test_output_time = time.time() - test_output_start
    print(f'测试输出计算耗时: {test_output_time:.4f}s')
    
    time_end = time.time()
    test_time = time_end - time_start
    print('Testing accurate is {:.4f}'.format(test_acc * 100,), '%')
    print('Testing time is {:.4f}'.format(test_time), 's')
    
    # # 绘制训练准确率和测试准确率随节点数量变化的曲线
    # plt.figure(figsize=(10, 6))
    # plt.plot(node_counts, train_accuracies, 'b-', marker='o', label='训练准确率')
    # plt.plot(node_counts, test_accuracies, 'r-', marker='x', label='测试准确率')
    # plt.xlabel('隐藏层节点数量')
    # plt.ylabel('准确率')
    # plt.title('准确率随节点数量的变化曲线')
    # plt.grid(True)
    # plt.legend()
    # plt.savefig('accuracy_vs_nodes.png')
    # plt.show()
    
    # 保存数据到CSV文件
    accuracy_data = {
        '节点数量': node_counts,
        '训练准确率': train_accuracies,
        '测试准确率': test_accuracies
    }
    import pandas as pd
    acc_df = pd.DataFrame(accuracy_data)
    acc_df.to_csv('accuracy_vs_nodes.csv', index=False)
    
    print('准确率曲线已保存为accuracy_vs_nodes.png')
    print('准确率数据已保存为accuracy_vs_nodes.csv')

    return train_acc, train_time, test_acc, test_time, W_opt, B_opt, OutputWeight

# 添加模型加载和测试函数
def load_and_test_model(model_path, test_x, test_y, model_index=0, time_steps=500):
    """
    加载保存的模型文件并在测试数据上进行评估
    
    参数:
    model_path: 模型文件路径
    test_x: 测试数据
    test_y: 测试标签
    model_index: 模型文件中的模型索引，默认为0（第一个模型）
    time_steps: 时间步长
    
    返回:
    test_acc: 测试准确率
    """
    import pickle
    
    # 加载模型文件
    with open(model_path, 'rb') as f:
        model_file_data = pickle.load(f)
    
    # 获取模型列表
    if 'models' in model_file_data:
        # 新格式：包含多个模型的文件
        models = model_file_data['models']
        if model_index >= len(models):
            raise ValueError(f"模型索引超出范围，文件中只有{len(models)}个模型")
        
        model_data = models[model_index]
        
        W_opt = model_data['W_opt']
        B_opt = model_data['B_opt']
        OutputWeight = model_data['OutputWeight']
        
        print(f"使用模型文件中的第{model_index+1}个模型（测试准确率: {model_data['test_acc']:.4f}）")
    else:
        # 旧格式：单个模型
        W_opt = model_file_data['W_opt']
        B_opt = model_file_data['B_opt']
        OutputWeight = model_file_data['OutputWeight']
    
    # 对测试数据进行脉冲编码
    test_x_spikes = poisson_encoding(test_x, time_steps)
    
    # 使用隐藏层计算
    tempH_test = lif_layer(test_x_spikes, W_opt, B_opt, time_steps)
    
    # 是否有直接连接
    if len(tempH_test[0]) != len(OutputWeight):  # 如果隐藏层输出维度与输出权重不匹配，则可能存在直接连接
        A_test = np.hstack([test_x, tempH_test])
    else:
        A_test = tempH_test
    
    # 计算输出
    test_output = np.dot(A_test, OutputWeight)
    test_acc = classification_accuracy(test_output, test_y)
    
    print(f'模型在测试集上的准确率: {test_acc:.4f}')
    return test_acc

# 添加加载最佳模型的辅助函数
def load_best_model_from_file(model_path, test_x, test_y, time_steps=500):
    """
    从模型文件中加载测试准确率最高的模型并进行评估
    
    参数:
    model_path: 模型文件路径
    test_x: 测试数据
    test_y: 测试标签
    time_steps: 时间步长
    
    返回:
    test_acc: 测试准确率
    best_model: 最佳模型数据
    """
    import pickle
    
    # 加载模型文件
    with open(model_path, 'rb') as f:
        model_file_data = pickle.load(f)
    
    # 获取模型列表
    if 'models' in model_file_data:
        models = model_file_data['models']
        # 按测试准确率排序
        models.sort(key=lambda x: x['test_acc'], reverse=True)
        # 获取最佳模型
        best_model = models[0]
        
        print(f"使用模型文件中测试准确率最高的模型（准确率: {best_model['test_acc']:.4f}）")
        
        return load_and_test_model(model_path, test_x, test_y, model_index=0, time_steps=time_steps), best_model
    else:
        # 旧格式：单个模型
        return load_and_test_model(model_path, test_x, test_y, time_steps=time_steps), model_file_data

def run_kfold_cross_validation(dataset_name, n_splits=5, dl=0, delta=1, c=2**-20, 
                           Lambdas=None, r=0.7, verbose=1, Lmax=45, tol=1e-2, 
                           Tmax=5, time_steps=2000, normalize=True, random_state=42,
                           num_runs=1):
    """
    在UCI数据集上进行K折交叉验证，并返回平均准确率和标准差
    
    参数:
    dataset_name: 数据集名称，如'iris', 'breast_cancer', 'ionosphere'等
    n_splits: 折叠数，默认为5
    dl, delta, c, Lambdas, r, verbose, Lmax, tol, Tmax, time_steps: SCN_SNN算法的参数
    normalize: 是否标准化特征，默认为True
    random_state: 随机种子，确保结果可复现
    num_runs: 每个折叠运行的次数，默认为1
    
    返回:
    avg_acc: 平均准确率
    std_acc: 准确率标准差
    """
    if Lambdas is None:
        Lambdas = [0.01, 0.05, 0.1, 0.5, 1, 5]
    
    # 准备K折交叉验证数据
    folds = image_process_copy.prepare_kfold_cross_validation_uci(
        dataset_name, n_splits, normalize
    )
    
    # 存储每个折叠的准确率
    fold_accuracies = []
    
    # 对每个折叠进行训练和测试
    for fold_idx, (X_train, y_train, X_test, y_test) in enumerate(folds):
        print(f"\n===== 第 {fold_idx+1}/{n_splits} 折 =====")
        
        fold_test_accs = []
        
        # 每个折叠运行num_runs次
        for run in range(num_runs):
            print(f"\n----- 运行 {run+1}/{num_runs} -----")
            
            # # 设置随机种子（折叠编号+运行编号+基础随机种子）
            # np.random.seed(random_state + fold_idx*100 + run)
            
            # 训练模型
            train_acc, train_time, test_acc, test_time, W_opt, B_opt, OutputWeight = SCN_classification_SNN(
                X_train, y_train, X_test, y_test, dl, delta, c, Lambdas, r, verbose, Lmax, tol,
                Tmax, time_steps=time_steps
            )
            
            # 记录测试准确率
            fold_test_accs.append(test_acc)
            
            print(f"训练准确率: {train_acc:.4f}, 测试准确率: {test_acc:.4f}")
            print(f"训练时间: {train_time:.4f}s, 测试时间: {test_time:.4f}s")
        
        # 计算当前折叠的平均准确率
        fold_avg_acc = np.mean(fold_test_accs)
        fold_accuracies.append(fold_avg_acc)
        
        print(f"\n第 {fold_idx+1} 折平均测试准确率: {fold_avg_acc:.4f}")
    
    # 计算所有折叠的平均准确率和标准差
    avg_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    
    # 打印每个折叠的准确率
    print("\n===== 交叉验证结果 =====")
    for i, acc in enumerate(fold_accuracies):
        print(f"第 {i+1} 折准确率: {acc:.4f}")
    
    print(f"\n平均准确率: {avg_acc:.4f} ± {std_acc:.4f}")
    
    # 创建结果目录
    import os
    result_dir = f'{dataset_name}_cv_results'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # 保存结果到CSV文件
    import pandas as pd
    result_data = {
        '数据集': dataset_name,
        '折叠数': n_splits,
        '运行次数': num_runs,
        '平均准确率': avg_acc,
        '准确率标准差': std_acc,
        '参数_dl': dl,
        '参数_delta': delta,
        '参数_c': c,
        '参数_Lambdas': str(Lambdas),
        '参数_r': r,
        '参数_Lmax': Lmax,
        '参数_tol': tol,
        '参数_Tmax': Tmax,
        '参数_time_steps': time_steps
    }
    result_df = pd.DataFrame([result_data])
    result_df.to_csv(os.path.join(result_dir, f'{dataset_name}_cv_results.csv'), index=False)
    
    # 创建更详细的结果数据
    detailed_data = []
    for i, acc in enumerate(fold_accuracies):
        detailed_data.append({
            '数据集': dataset_name,
            '折叠编号': i+1,
            '准确率': acc
        })
    detailed_df = pd.DataFrame(detailed_data)
    detailed_df.to_csv(os.path.join(result_dir, f'{dataset_name}_detailed_cv_results.csv'), index=False)
    
    # 绘制准确率条形图
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, n_splits+1), fold_accuracies, color='skyblue')
    plt.axhline(y=avg_acc, color='red', linestyle='-', label=f'平均准确率: {avg_acc:.4f}')
    plt.fill_between([0.5, n_splits+0.5], [avg_acc-std_acc, avg_acc-std_acc], 
                     [avg_acc+std_acc, avg_acc+std_acc], color='red', alpha=0.2)
    plt.xlabel('折叠编号')
    plt.ylabel('准确率')
    plt.title(f'{dataset_name}数据集 {n_splits}折交叉验证结果')
    plt.xticks(range(1, n_splits+1))
    plt.ylim(0, 1.1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f'{dataset_name}_cv_accuracy.png'))
    plt.show()
    
    return avg_acc, std_acc

if __name__ == '__main__':
    # 在UCI数据集上进行交叉验证
    print('--------------------UCI数据集交叉验证-----------------------------')
    
    # 设置参数
    dl = 1  # with direct links
    delta = 1  # number of hidden node addition each step
    Lmax = 20  # the maximum number of hidden nodes
    Tmax = 5
    tol = 1e-2  # the tolerance error
    r = 0.7
    Lambdas = [0.01, 0.05, 0.1, 0.5, 1, 5]  # scope sequence
    verbose = 1  # print frequency
    c = 2 ** -20  # l2 Regularization coefficient
    time_steps = 500
    
    # 运行交叉验证
    # dataset_names = ['iris', 'breast_cancer', 'ionosphere']
    dataset_names = ['iris']
    
    for dataset_name in dataset_names:
        # 为确保可重复性，设置全局随机种子
        np.random.seed(5)
        print(f'\n\n========== 数据集: {dataset_name} ==========')
        avg_acc, std_acc = run_kfold_cross_validation(
            dataset_name=dataset_name,
            n_splits=5,
            dl=dl,
            delta=delta,
            c=c,
            Lambdas=Lambdas,
            r=r,
            verbose=verbose,
            Lmax=Lmax,
            tol=tol,
            Tmax=Tmax,
            time_steps=time_steps,
            normalize=True,
            # random_state=42,
            num_runs=1  # 每个折叠运行一次
        )
        
        print(f'{dataset_name}数据集 5折交叉验证准确率: {avg_acc:.4f} ± {std_acc:.4f}')
    
