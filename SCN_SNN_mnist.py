# -*- coding: utf-8 -*-
import numpy as np
from math import sqrt
from sklearn import preprocessing
from scipy import linalg as LA
import time
import matplotlib.pyplot as plt
import image_process_copy
from tqdm import tqdm
from SCN_SNN import *

if __name__ == '__main__': 
    # 加载MNIST数据集
    train_data, train_label, test_data, test_label = image_process_copy.load_and_preprocess_mnist()
    dl = 0  # with direct links
    Lmax = 20  # the maximum number of hidden nodes
    #######################################################
    delta = 1  # number of hidden node addition each step
    Tmax = 5
    tol = 1e-2  # the tolerance error
    r = 0.7
    # Lambdas = [0.01, 0.05, 0.1, 0.5, 1, 5]  # scope sequence
    Lambdas = [10, 20, 50, 100, 200, 500, 1000]  # scope sequence
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

    best_seed = 0
    best_acc = 0
    acc_list = []
    best_run_results = []  # 初始化保存最佳结果的列表
    for seed in range(1, 10):
        np.random.seed(seed)
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

            # # 绘制各个权重参数的pdf
            # plt.figure(figsize=(10, 6))
            # plt.subplot(1, 2, 1)
            # plt.hist(W_opt.flatten(), bins=100, density=True, alpha=0.75)
            # plt.title('Weight Distribution')
            # plt.xlabel('Weight Value')
            # plt.ylabel('Frequency')
            # plt.subplot(1, 2, 2)
            # plt.hist(B_opt.flatten(), bins=100, density=True, alpha=0.75)
            # plt.title('Bias Distribution')
            # plt.xlabel('Bias Value')
            # plt.ylabel('Frequency')
            # plt.savefig('weight_bias_distribution.png')
            # plt.show()

        # 计算平均准确率和时间
        avg_train_acc = total_train_acc / num_runs
        avg_train_time = total_train_time / num_runs
        avg_test_acc = total_test_acc / num_runs
        avg_test_time = total_test_time / num_runs
        print(f'平均训练准确率: {avg_train_acc:.4f}')
        print(f'平均训练时间: {avg_train_time:.4f}秒')
        print(f'平均测试准确率: {avg_test_acc:.4f}')
        print(f'平均测试时间: {avg_test_time:.4f}秒')
        
        # 按测试准确率对结果排序（降序）
        run_results.sort(key=lambda x: x['test_acc'], reverse=True)
        
        # 保存每个种子的所有模型
        import os
        import pickle
        
        # 创建保存模型的目录
        model_dir = 'wine_seed_models'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        # 将当前种子的所有模型保存在一个文件中
        model_file = os.path.join(model_dir, f'seed_{seed}_models.pkl')
        with open(model_file, 'wb') as f:
            # 保存为包含所有模型的列表
            pickle.dump({
                'models': run_results,
                'seed': seed,
                'avg_test_acc': avg_test_acc
            }, f)
        print(f'已将种子{seed}的所有模型保存至: {model_file}')
        
        # 保存结果到CSV文件
        result_data = {
            '模型': 'SCN_分类_MNIST',
            '运行次数': num_runs,
            '平均训练准确率': avg_train_acc,
            '平均训练时间': avg_train_time,
            '平均测试准确率': avg_test_acc,
            '平均测试时间': avg_test_time
        }
        import pandas as pd
        result_df = pd.DataFrame([result_data])
        result_df.to_csv('SCN_分类_MNIST_结果.csv', index=False)

        acc_list.append(avg_test_acc)
        if avg_test_acc > best_acc:
            best_acc = avg_test_acc
            best_seed = seed
            best_run_results = run_results.copy()  # 保存当前种子的最佳结果

    print(f'最佳种子: {best_seed}, 最佳准确率: {best_acc:.4f}')

    # 保存所有种子中最好的模型
    best_models_dir = 'wine_best_models'
    if not os.path.exists(best_models_dir):
        os.makedirs(best_models_dir)

    # 将最佳种子的所有模型保存在一个文件中
    best_models_file = os.path.join(best_models_dir, f'best_seed_{best_seed}_models.pkl')
    with open(best_models_file, 'wb') as f:
        # 保存为包含所有模型的列表
        pickle.dump({
            'models': best_run_results,
            'best_seed': best_seed,
            'best_avg_acc': best_acc
        }, f)
    print(f'已将最佳种子({best_seed})的所有模型保存至: {best_models_file}')

    plt.figure(figsize=(10, 6))
    plt.plot(acc_list, label='测试准确率')
    plt.xlabel('种子')
    plt.ylabel('准确率')
    plt.title('测试准确率随种子变化曲线')
    plt.legend()
    plt.show()
    
