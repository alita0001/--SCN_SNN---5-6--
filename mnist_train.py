from SCN_SNN import *
import umap

# 神经网络参数设置 - 为每个数据集定义专门的参数
# 数据集特定参数
# 定义数据集名称列表
# dataset_names = ['mnist', 'fashion_mnist']
dataset_names = ['mnist']

# 为每个数据集设置固定的随机种子，确保结果可重现
dataset_seed = {
    'mnist': 1,
    'fashion_mnist': 1
}
dataset_params = {
    'mnist': {
        'dl': 1,  # 是否使用直接连接
        'Lmax': 200,
        'delta': 1,
        'Tmax': 1,
        'r': 0.7,
        'Lambdas': [0.01, 0.05, 0.1, 0.5, 1, 5],
        'c': 2 ** -20,
        'verbose': 1,  # 打印频率
        'tol': 1e-2,  # 容差误差
        'time_steps': 10
    },
    'fashion_mnist': {
        'dl': 1,  # 是否使用直接连接
        'Lmax': 20,
        'delta': 1,
        'Tmax': 5,
        'r': 0.7,
        'Lambdas': [0.01, 0.05, 0.1, 0.5, 1, 5],
        'c': 2 ** -20,
        'verbose': 1,  # 打印频率
        'tol': 1e-2,  # 容差误差
        'time_steps': 100
    }
}

# 加载数据集
def load_data(dataset_name):
    if dataset_name == 'mnist':
        train_data, train_label, test_data, test_label = image_process_copy.load_and_preprocess_mnist()
    elif dataset_name == 'fashion_mnist':
        train_data, train_label, test_data, test_label = image_process_copy.load_and_preprocess_fashion_mnist()
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
        
    # 返回加载的数据
    return train_data, train_label, test_data, test_label

# UMAP降维预处理函数
def apply_umap_preprocessing(train_data, test_data, n_components=2, n_neighbors=15, min_dist=0.1):
    """
    使用UMAP对数据进行降维处理
    
    参数:
    train_data: 训练数据，形状为 [n_samples, n_features]
    test_data: 测试数据，形状为 [n_samples, n_features]
    n_components: 降维后的维度
    n_neighbors: UMAP参数，用于控制局部邻域大小
    min_dist: UMAP参数，用于控制嵌入点之间的最小距离
    
    返回:
    train_data_umap: 降维后的训练数据
    test_data_umap: 降维后的测试数据
    umap_reducer: 训练好的UMAP模型
    """
    print(f"使用UMAP将数据从{train_data.shape[1]}维降至{n_components}维...")
    
    # 创建UMAP对象
    umap_reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42
    )
    
    # 对训练数据进行降维
    start_time = time.time()
    train_data_umap = umap_reducer.fit_transform(train_data)
    fit_time = time.time() - start_time
    print(f"UMAP拟合训练数据耗时: {fit_time:.2f}秒")
    
    # 对测试数据进行转换
    start_time = time.time()
    test_data_umap = umap_reducer.transform(test_data)
    transform_time = time.time() - start_time
    print(f"UMAP转换测试数据耗时: {transform_time:.2f}秒")
    
    print(f"降维后的数据形状: 训练数据 {train_data_umap.shape}, 测试数据 {test_data_umap.shape}")
    
    # 如果降维后的维度为2或3，可以绘制降维结果
    if n_components == 2:
        plt.figure(figsize=(10, 8))
        plt.scatter(train_data_umap[:, 0], train_data_umap[:, 1], s=5, alpha=0.5)
        plt.title('UMAP 2D投影')
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.show()
    elif n_components == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(train_data_umap[:, 0], train_data_umap[:, 1], train_data_umap[:, 2], s=5, alpha=0.5)
        ax.set_title('UMAP 3D投影')
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.set_zlabel('UMAP3')
        plt.show()
    
    return train_data_umap, test_data_umap, umap_reducer

def main():
    for dataset_name, seed in zip(dataset_names, dataset_seed):
        train_data, train_label, test_data, test_label = load_data(dataset_name)

        # 使用UMAP对数据进行降维
        train_data, test_data, umap_reducer = apply_umap_preprocessing(train_data, test_data,
                                                                        # 选择较好的UMAP参数组合
                                                                        # n_components: 降维后的维度，2维便于可视化
                                                                        # n_neighbors: 较大值(15-50)保留全局结构，较小值(5-15)保留局部结构
                                                                        # min_dist: 较小值(0.001-0.1)使聚类更紧密，较大值(0.5-0.8)使分布更均匀
                                                                        n_components=50, 
                                                                        n_neighbors=15,  # 增大以更好地保留全局结构
                                                                        min_dist=0.1)   # 调小以使聚类更紧密
        # 保存拟合之后的数据
        np.save(f'{dataset_name}_train_data_umap.npy', train_data_umap)
        np.save(f'{dataset_name}_test_data_umap.npy', test_data_umap)
        # 加载拟合之后的数据
        train_data_umap = np.load(f'{dataset_name}_train_data_umap.npy')
        test_data_umap = np.load(f'{dataset_name}_test_data_umap.npy')
        print(f"训练数据集: {dataset_name}")
        np.random.seed(dataset_seed[dataset_name])
        num_runs = 1
        total_train_acc = 0
        total_train_time = 0
        total_test_acc = 0
        total_test_time = 0
            
        # 创建一个列表来跟踪每次运行的结果
        run_results = []
        
        for i in range(num_runs):
            train_acc, train_time, test_acc, test_time, W_opt, B_opt, OutputWeight = SCN_classification_SNN(train_data, train_label, test_data, test_label, **dataset_params[dataset_name])
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
        model_dir = f'uci_train_models/{dataset_name}'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        # 将当前种子的所有模型保存在一个文件中
        model_file = os.path.join(model_dir, f'{dataset_name}_models.pkl')
        with open(model_file, 'wb') as f:
            # 保存为包含所有模型的列表和数据集参数
            pickle.dump({
                'models': run_results,
                'avg_test_acc': avg_test_acc,
                'test_acc_std': test_acc_std,
                'avg_train_acc': avg_train_acc,
                'train_acc_std': train_acc_std,
                'dataset_params': dataset_params[dataset_name],  # 保存数据集参数
                'umap_reducer': umap_reducer,  # 保存UMAP模型
                'avg_test_time': avg_test_time,
                'avg_train_time': avg_train_time
            }, f)
        print(f'所有模型和数据集参数保存至: {model_file}')
        
        # 保存结果到CSV文件
        result_data = {
            '模型': f'{dataset_name}',
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
        result_df.to_csv(f'./{model_dir}/{dataset_name}_结果.csv', index=False)

        # 打印训练准确率汇总表
        print("\n训练准确率汇总表")
        print(f"{'数据集名称':<15}{'第1次':<10}{'第2次':<10}{'第3次':<10}{'第4次':<10}{'第5次':<10}{'平均准确率':<12}{'标准差':<10}{'参数数量':<10}")
        print('-' * 110)
        
        # 准备每次测试的准确率字符串
        acc_strings = []
        for i in range(num_runs):
            if i < len(run_results):
                acc_strings.append(f"{run_results[i]['test_acc']*100:.2f}%")
            else:
                acc_strings.append("--")
        
        # 计算参数数量
        total_params = 0
        if len(run_results) > 0:
            best_model = run_results[0]  # 取第一个模型（已按准确率排序）
            w_params = best_model['W_opt'].size  # W_opt的参数数量
            b_params = best_model['B_opt'].size  # B_opt的参数数量
            output_params = best_model['OutputWeight'].size  # OutputWeight的参数数量
            total_params = w_params + b_params + output_params  # 总参数数量
        
        # 打印一行数据
        print(f"{dataset_name:<15}", end="")
        for acc_str in acc_strings:
            print(f"{acc_str:<10}", end="")
        print(f"{avg_test_acc*100:.2f}%{'':<8}{test_acc_std:.2f}{'':<6}{total_params}")
        
        print('-' * 110)

    # 打印所有数据集训练结果的汇总表
    print("\n所有数据集训练结果汇总表")
    print(f"{'数据集名称':<15}{'平均训练准确率':<15}{'训练准确率标准差':<15}{'平均测试准确率':<15}{'测试准确率标准差':<15}{'平均训练时间(s)':<15}{'平均测试时间(s)':<15}{'参数数量':<10}")
    print('-' * 120)
    
    # 收集所有数据集的结果
    all_results = []
    for dataset_name in dataset_names:
        model_file = f'./{model_dir}/{dataset_name}_models.pkl'
        print(f"{dataset_name}模型文件: {model_file}")
        if os.path.exists(model_file):
            with open(model_file, 'rb') as f:
                saved_data = pickle.load(f)
                
            # 计算参数数量
            total_params = 0
            if 'run_results' in saved_data and len(saved_data['run_results']) > 0:
                best_model = saved_data['run_results'][0]
                w_params = best_model['W_opt'].size
                b_params = best_model['B_opt'].size
                output_params = best_model['OutputWeight'].size
                total_params = w_params + b_params + output_params
            
            # 打印一行数据
            print(f"{dataset_name:<15}{saved_data['avg_train_acc']*100:.2f}%{'':<10}{saved_data['train_acc_std']*100:.2f}%{'':<10}{saved_data['avg_test_acc']*100:.2f}%{'':<10}{saved_data['test_acc_std']*100:.2f}%{'':<10}{saved_data['avg_train_time']:<15.2f}{saved_data['avg_test_time']:<15.2f}{total_params}")
        else:
            print(f"{dataset_name:<15} 没有训练结果")
    print('-' * 120)

# 执行主函数
if __name__ == '__main__':
    main()
    # 代码执行结束后，发送消息通知
    import os
    os.system('msg %username% "代码执行完成!"')

