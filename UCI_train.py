from SCN_SNN import *

# 神经网络参数设置 - 为每个数据集定义专门的参数
# 数据集特定参数
# 定义数据集名称列表
dataset_names = ['iris', 'hepatitis', 'breast_cancer', 'ionosphere', 'wine']
# dataset_names = ['hepatitis']

# 为每个数据集设置固定的随机种子，确保结果可重现
dataset_seed = {
    'iris': 1,
    'hepatitis': 9,
    'breast_cancer': 7,
    'ionosphere': 5,
    'wine': 5
}
dataset_params = {
    'iris': {
        'dl': 1,  # 是否使用直接连接
        'Lmax': 20,
        'delta': 1,
        'Tmax': 5,
        'r': 0.7,
        'Lambdas': [0.01, 0.05, 0.1, 0.5, 1, 5],
        'c': 2 ** -20,
        'verbose': 1,  # 打印频率
        'tol': 1e-2,  # 容差误差
        'time_steps': 500
    },
    'hepatitis': {
        'dl': 0,
        'Lmax': 20,
        'delta': 1,
        'Tmax': 5,
        'r': 0.7,
        'Lambdas': [10, 20, 50, 100, 200, 500, 1000],
        'c': 2 ** -20,
        'verbose': 1,
        'tol': 1e-2,
        'time_steps': 500
    },
    'breast_cancer': {
        'dl': 1,
        'Lmax': 20,
        'delta': 5,
        'Tmax': 5,
        'r': 0.7,
        'Lambdas': [0.01, 0.05, 0.1, 0.5, 1, 5],
        'c': 2 ** -20,
        'verbose': 1,
        'tol': 1e-2,
        'time_steps': 500
    },
    'ionosphere': {
        'dl': 0,
        'Lmax': 45,
        'delta': 1,
        'Tmax': 5,
        'r': 0.7,
        'Lambdas': [0.01, 0.05, 0.1, 0.5, 1, 5],
        'c': 2 ** -20,
        'verbose': 1,
        'tol': 1e-2,
        'time_steps': 2000
    },
    'wine': {
        'dl': 1,
        'Lmax': 10,
        'delta': 1,
        'Tmax': 5,
        'r': 0.7,
        'Lambdas': [0.01, 0.05, 0.1, 0.5, 1, 5],
        'c': 2 ** -20,
        'verbose': 1,
        'tol': 1e-2,
        'time_steps': 500
    }
}

# 加载数据集
def load_data(dataset_name):
    if dataset_name == 'iris':
        train_data, train_label, test_data, test_label = image_process_copy.load_and_preprocess_iris(test_size=0.2, shuffle=True, normalize=False)
    elif dataset_name == 'hepatitis':
        train_data, train_label, test_data, test_label = image_process_copy.load_and_preprocess_hepatitis(test_size=0.5, shuffle=True, normalize=False)
    elif dataset_name == 'breast_cancer':
        train_data, train_label, test_data, test_label = image_process_copy.load_and_preprocess_breast_cancer(test_size=0.2, shuffle=True, normalize=False)
    elif dataset_name == 'ionosphere':
        train_data, train_label, test_data, test_label = image_process_copy.load_and_preprocess_ionosphere(test_size=0.2, shuffle=True, normalize=False)
    elif dataset_name == 'wine':
        train_data, train_label, test_data, test_label = image_process_copy.load_and_preprocess_wine(test_size=0.2, shuffle=True, normalize=False)
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
        
    # 返回加载的数据
    return train_data, train_label, test_data, test_label

def main():
    for dataset_name, seed in zip(dataset_names, dataset_seed):
        train_data, train_label, test_data, test_label = load_data(dataset_name)
        
        print(f"训练数据集: {dataset_name}")
        np.random.seed(dataset_seed[dataset_name])
        num_runs = 5
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
                'dataset_params': dataset_params[dataset_name]  # 保存数据集参数
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
