# -*- coding: utf-8 -*-
import numpy as np
from math import sqrt
from sklearn import preprocessing
from scipy import linalg as LA
import time
import matplotlib.pyplot as plt
import image_process_copy

# 设置随机种子，确保每次运行结果一致
np.random.seed(1)  # 42是一个常用的随机种子值，可以更改为任意整数

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

if __name__ == '__main__':
    import scipy.io as scio
    # ---------------回归任务:归一化后的高次复合函数---------------
    # Dataset = scio.loadmat('Demo_data.mat')
    # train_data = Dataset['P1']
    # train_label = Dataset['T1']
    # test_data = Dataset['P2']
    # test_label = Dataset['T2']

    # train_data = np.linspace(-1, 1, 800).reshape(-1, 1)
    # train_label = train_data ** 3  # + np.random.normal(0, 0.1, train_data.shape)
    # # train_label = np.sin(train_data)
    # print(train_data.shape, train_label.shape)
    # test_data = np.linspace(-1, 1, 800).reshape(-1, 1)
    # test_label = test_data ** 3  # + np.random.normal(0, 0.1, test_data.shape)
    # # test_label = np.sin(test_data)
    # print(test_data.shape, test_label.shape)

    # train_data = np.linspace(0, 1, 800).reshape(-1, 1)
    # term1 = 0.2 * np.exp(-np.square(10 * train_data - 4))
    # term2 = 0.5 * np.exp(-np.square(80 * train_data - 40))
    # term3 = 0.3 * np.exp(-np.square(80 * train_data - 20))
    # # train_label = train_data ** 3  # + np.random.normal(0, 0.1, train_data.shape)
    # train_label = term1 + term2 + term3
    # print(train_data.shape, train_label.shape)
    # test_data = np.linspace(0, 1, 800).reshape(-1, 1)
    # term1 = 0.2 * np.exp(-np.square(10 * test_data - 4))
    # term2 = 0.5 * np.exp(-np.square(80 * test_data - 40))
    # term3 = 0.3 * np.exp(-np.square(80 * test_data - 20))
    # # train_label = train_data ** 3  # + np.random.normal(0, 0.1, train_data.shape)
    # test_label = term1 + term2 + term3

    # # parameter setting
    # dl = 1  # with direct links
    # Lambda = 150  # the assignment scope of random parameters
    # Lmax = 20  # the maximum number of hidden nodes
    # delta = 1  # number of hidden node addition each step
    # Tmax = 2
    # tol = 1e-2  # the tolerance error
    # r = 0.8
    # Lambdas = [0.1, 1, 15, 20, 25, 300]  # scope sequence
    # verbose = 5  # print frequency
    # c = 2 ** -30  # l2 Regularization coefficient
    # Regression tasks
    # print('-------------------RVFLN---------------------------')
    # RVFLN_regression(train_data, train_label, test_data, test_label, dl, c, Lambda, Lmax)
    # print('-------------------IRVFLN---------------------------')
    # IRVFLN_regression(train_data, train_label, test_data, test_label, dl, delta, c, Lambda, verbose, Lmax, tol)
    # print('--------------------SCN_SNN-----------------------------')
    # SCN_regression_SNN(train_data, train_label, test_data, test_label, dl, delta, c, Lambdas, r, verbose, Lmax, tol, Tmax)

    # ------------------------------分类任务: MNIST-------------------------------------
    # # 处理CIFAR-10
    # cifar10_train_data, cifar10_train_labels, cifar10_test_data, cifar10_test_labels = image_process_copy.load_and_preprocess_cifar10()
    
    # # 处理CIFAR-100
    # cifar100_train_data, cifar100_train_labels, cifar100_test_data, cifar100_test_labels = image_process_copy.load_and_preprocess_cifar100()
    
    # # 打印形状信息
    # print("CIFAR-10 Shapes:")
    # print("Train data:", cifar10_train_data.shape)  # (50000, 3072)
    # print("Train labels:", cifar10_train_labels.shape)  # (50000, 10)
    # print("Test data:", cifar10_test_data.shape)  # (10000, 3072)
    # print("Test labels:", cifar10_test_labels.shape)  # (10000, 10)
    # print(cifar100_train_data[1])
    
    # print("\nCIFAR-100 Shapes:")
    # print("Train data:", cifar100_train_data.shape)  # (50000, 3072)
    # print("Train labels:", cifar100_train_labels.shape)  # (50000, 100)
    # print("Test data:", cifar100_test_data.shape)  # (10000, 3072)
    # print("Test labels:", cifar100_test_labels.shape)  # (10000, 100)
    
    
    # 加载MNIST数据集
    train_data, train_label, test_data, test_label = image_process_copy.load_and_preprocess_mnist()

    # 加载iris数据集
    train_data, train_label, test_data, test_label = image_process_copy.load_and_preprocess_pima_diabetes(test_size=0.2, shuffle=True, normalize=False)
    
    train_data, train_label, test_data, test_label = image_process_copy.load_and_preprocess_iris(test_size=0.2, shuffle=True, normalize=False)
    # parameter setting
    # train_data, train_label, test_data, test_label = image_process_copy.load_and_preprocess_breast_cancer(test_size=0.2, shuffle=True, normalize=False)
    
    dl = 1  # with direct links
    Lambda = 1  # the assignment scope of random parameters
    Lmax = 20  # the maximum number of hidden nodes
    #######################################################
    delta = 1  # number of hidden node addition each step
    Tmax = 5
    tol = 1e-2  # the tolerance error
    r = 0.7
    Lambdas = [0.01, 0.05, 0.1, 0.5, 1, 5]  # scope sequence
    # Lambdas = [10, 20, 50, 100, 200, 500, 1000]  # scope sequence
    verbose = 1  # print frequency
    c = 2 ** -20  # l2 Regularization coefficient
    # Classification tasks
    # print('-------------------RVFLN---------------------------')
    # RVFLN_classification(train_data, train_label, test_data, test_label, dl, c, Lambda, Lmax)
    # print('-------------------IRVFLN---------------------------')
    # IRVFLN_classification(train_data, train_label, test_data, test_label, dl, delta, c, Lambda, verbose, Lmax, tol)
    # print('--------------------SCN_分类_CIFAR10-----------------------------')
    # SCN_classification_SNN(cifar10_train_data, cifar10_train_labels, cifar10_test_data, cifar10_test_labels, dl, delta, c, Lambdas, r, verbose, Lmax, tol,
    #                    Tmax)
    print('--------------------SCN_分类_MNIST-----------------------------')
    # 运行5次，计算模型训练的平均准确率

    np.random.seed(1)
    num_runs = 5
    total_train_acc = 0
    total_train_time = 0
    total_test_acc = 0
    total_test_time = 0
        
    # 创建一个列表来跟踪每次运行的结果
    run_results = []
    
    for i in range(num_runs):
        train_acc, train_time, test_acc, test_time, W_opt, B_opt, OutputWeight = SCN_classification_SNN(train_data, train_label, test_data, test_label, dl, delta, c, Lambdas, r, verbose, Lmax, tol,
                    Tmax, time_steps=500)
        total_train_acc += train_acc
        total_train_time += train_time
        total_test_acc += test_acc
        total_test_time += test_time
        
        # 保存每次运行的模型参数和准确率
        run_results.append({
            'run_id': i,
            'test_acc': test_acc,
            'train_acc': train_acc,
            'W_opt': W_opt,
            'B_opt': B_opt,
            'OutputWeight': OutputWeight
        })

    # 计算平均准确率和时间
    avg_train_acc = total_train_acc / num_runs
    avg_train_time = total_train_time / num_runs
    avg_test_acc = total_test_acc / num_runs
    avg_test_time = total_test_time / num_runs
    
    # 计算准确率的标准差
    train_acc_list = [result['train_acc']*100 for result in run_results]
    test_acc_list = [result['test_acc']*100 for result in run_results]
    train_acc_std = np.std(train_acc_list)
    test_acc_std = np.std(test_acc_list)
    
    # 打印每次运行的准确率
    for i, result in enumerate(run_results):
        print(f'第{i+1}次运行: 训练准确率={result["train_acc"]:.4f}, 测试准确率={result["test_acc"]:.4f}')

    print(f'平均训练准确率: {avg_train_acc:.4f} ± {train_acc_std:.4f}')
    print(f'平均训练时间: {avg_train_time:.4f}秒')
    print(f'平均测试准确率: {avg_test_acc:.4f} ± {test_acc_std:.4f}')
    print(f'平均测试时间: {avg_test_time:.4f}秒')
    
    # 按测试准确率对结果排序（降序）
    run_results.sort(key=lambda x: x['test_acc'], reverse=True)
    
    # 保存每个种子的所有模型
    import os
    import pickle
    
    # 创建保存模型的目录
    model_dir = 'uci_models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    # 将当前种子的所有模型保存在一个文件中
    model_file = os.path.join(model_dir, f'iris_models.pkl')
    with open(model_file, 'wb') as f:
        # 保存为包含所有模型的列表
        pickle.dump({
            'models': run_results,
            'avg_test_acc': avg_test_acc,
            'test_acc_std': test_acc_std,
            'avg_train_acc': avg_train_acc,
            'train_acc_std': train_acc_std
        }, f)
    print(f'所有模型保存至: {model_file}')
    
    # 保存结果到CSV文件
    result_data = {
        '模型': 'SCN_分类_MNIST',
        '运行次数': num_runs,
        '平均训练准确率': avg_train_acc,
        '训练准确率标准差': train_acc_std,
        '平均训练时间': avg_train_time,
        '平均测试准确率': avg_test_acc,
        '测试准确率标准差': test_acc_std,
        '平均测试时间': avg_test_time
    }
    import pandas as pd
    result_df = pd.DataFrame([result_data])
    result_df.to_csv('SCN_分类_MNIST_结果.csv', index=False)

    
