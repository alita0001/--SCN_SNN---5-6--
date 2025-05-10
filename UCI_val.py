import image_process_copy
import numpy as np
from SCN_SNN import SCN_classification_SNN, poisson_encoding, lif_layer, classification_accuracy
import pickle
import os

# 添加模型加载和测试函数
def load_and_test_model(model_path, test_x, test_y, model_index=0):
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
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 '{model_path}' 不存在")
    
    try:
        # 加载模型文件
        with open(model_path, 'rb') as f:
            model_file_data = pickle.load(f)
    except Exception as e:
        raise Exception(f"加载模型文件 '{model_path}' 时出错: {e}")
    
    # 获取模型列表
    if 'models' in model_file_data:
        # 新格式：包含多个模型的文件
        models = model_file_data['models']
        if model_index >= len(models):
            raise ValueError(f"模型索引超出范围，文件中只有{len(models)}个模型")
        
        model_data = models[model_index]
        
        # 检查必要的模型参数是否存在
        required_keys = ['W_opt', 'B_opt', 'OutputWeight']
        for key in required_keys:
            if key not in model_data:
                raise KeyError(f"模型中缺少必要参数: {key}")
        
        W_opt = model_data['W_opt']
        B_opt = model_data['B_opt']
        OutputWeight = model_data['OutputWeight']
        
        print(f"使用模型文件中的第{model_index+1}个模型（测试准确率: {model_data.get('test_acc', 'N/A')}）")
    else:
        # 旧格式：单个模型
        required_keys = ['W_opt', 'B_opt', 'OutputWeight']
        for key in required_keys:
            if key not in model_file_data:
                raise KeyError(f"模型中缺少必要参数: {key}")
        
        W_opt = model_file_data['W_opt']
        B_opt = model_file_data['B_opt']
        OutputWeight = model_file_data['OutputWeight']
        
        print(f"使用单一模型文件（旧格式）")
    
    try:
        if 'dataset_params' in model_file_data:
            time_steps = model_file_data['dataset_params']['time_steps']
        else:
            time_steps = 500
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
        # 计算并返回权重参数的数量和测试准确率
        w_params = W_opt.size  # W_opt的参数数量
        b_params = B_opt.size  # B_opt的参数数量
        output_params = OutputWeight.size  # OutputWeight的参数数量
        total_params = w_params + b_params + output_params  # 总参数数量
        
        print(f'模型参数数量: 总计 {total_params} 个参数')
        print(f'  - 输入权重 (W_opt): {w_params} 个参数')
        print(f'  - 偏置 (B_opt): {b_params} 个参数')
        print(f'  - 输出权重 (OutputWeight): {output_params} 个参数')
        
        return test_acc, total_params
    except Exception as e:
        raise Exception(f"模型评估过程中出错: {e}")

def load_data(dataset_name):
    if dataset_name == 'iris':
        train_data, train_label, test_data, test_label = image_process_copy.load_and_preprocess_iris(test_size=0.3, shuffle=True, normalize=False)
    elif dataset_name == 'hepatitis':
        train_data, train_label, test_data, test_label = image_process_copy.load_and_preprocess_hepatitis(test_size=0.3, shuffle=True, normalize=False)
    elif dataset_name == 'breast_cancer':
        train_data, train_label, test_data, test_label = image_process_copy.load_and_preprocess_breast_cancer(test_size=0.3, shuffle=True, normalize=False)
    elif dataset_name == 'ionosphere':
        train_data, train_label, test_data, test_label = image_process_copy.load_and_preprocess_ionosphere(test_size=0.3, shuffle=True, normalize=False)
    elif dataset_name == 'wine':
        train_data, train_label, test_data, test_label = image_process_copy.load_and_preprocess_wine(test_size=0.3, shuffle=True, normalize=False)
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
        
    # 返回加载的数据
    return train_data, train_label, test_data, test_label

def main():
    dataset_names = ['iris', 'hepatitis', 'breast_cancer', 'ionosphere', 'wine']
    
    dataset_acc_dict = {}

    dataset_params_dict = {}

    for dataset_name in dataset_names:
        dataset_acc_dict[dataset_name] = []
        dataset_params_dict[dataset_name] = 0
        print(f"\n{'='*50}")
        print(f"处理数据集: {dataset_name}")
        print(f"{'='*50}")
        
        try:
            # 加载数据集
            train_data, train_label, test_data, test_label = load_data(dataset_name)
            print(f"数据集: {dataset_name}")
            print(f"训练集数据形状: {train_data.shape}")
            print(f"训练集标签形状: {train_label.shape}")
            print(f"测试集数据形状: {test_data.shape}")
            print(f"测试集标签形状: {test_label.shape}")

            # 设置模型文件路径
            root_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(root_dir, 'uci_train_models', f'{dataset_name}', f'{dataset_name}_models.pkl')
            
            # 尝试加载和测试模型
            try:
                for model_idx in range(5):  # 修改为从0到4，与下面的显示对应
                    try:
                        test_acc,dataset_params_dict[dataset_name] = load_and_test_model(model_path, test_data, test_label, model_idx)
                        dataset_acc_dict[dataset_name].append(test_acc)
                    except Exception as e:
                        print(f"加载模型 {model_idx} 失败: {e}")
                
                if dataset_acc_dict[dataset_name]:  # 如果至少有一个模型成功加载并测试
                    # 计算平均准确率和标准差
                    average_test_acc = np.mean(dataset_acc_dict[dataset_name])
                    std_test_acc = np.std(dataset_acc_dict[dataset_name]*100)
                    
                    # 打印每次的准确率
                    for i, acc in enumerate(dataset_acc_dict[dataset_name]):
                        print(f"第{i+1}次测试准确率: {acc*100:.2f}%")
                    
                    print(f"\n平均测试准确率: {average_test_acc*100:.2f}%，标准差: {std_test_acc*100:.2f}%")
                else:
                    print(f"警告: 没有成功加载任何模型进行测试")
            except FileNotFoundError:
                print(f"错误: 找不到模型文件 '{model_path}'")
            except Exception as e:
                print(f"测试模型时发生错误: {e}")
        
        except Exception as e:
            print(f"处理数据集 '{dataset_name}' 时发生错误: {e}")
            continue
    print(f"{'='*50}")
    print("所有数据集处理完成")
    print(f"{'='*50}")
    # 打印所有数据集的测试准确率表格
    print("\n数据集测试准确率汇总表")
    print(f"{'数据集名称':<10}{'第1次':<8}{'第2次':<8}{'第3次':<8}{'第4次':<8}{'第5次':<8}{'平均准确率':<9}{'标准差':<10}{'参数数量':<10}")
    print('-' * 100)
    
    for dataset_name in dataset_names:
        if not dataset_acc_dict[dataset_name]:
            print(f"{dataset_name:<15}{'无数据':<55}")
            continue
            
        # 准备每次测试的准确率字符串
        acc_strings = []
        for i in range(5):
            if i < len(dataset_acc_dict[dataset_name]):
                acc_strings.append(f"{dataset_acc_dict[dataset_name][i]*100:.2f}%")
            else:
                acc_strings.append("--")
        
        # 计算平均值和标准差
        avg_acc = np.mean([acc*100 for acc in dataset_acc_dict[dataset_name]])
        std_acc = np.std([acc*100 for acc in dataset_acc_dict[dataset_name]])
        
        # 打印一行数据
        print(f"{dataset_name:<15}", end="")
        for acc_str in acc_strings:
            print(f"{acc_str:<10}", end="")
        print(f"{avg_acc:.2f}%{'':<8}{std_acc:.2f}%{'':<10}{dataset_params_dict[dataset_name]:<10}")
    
    print('-' * 100)

if __name__ == '__main__':
    main()

    # 代码执行结束后，发送消息通知
    import os
    os.system('msg %username% "代码执行完成"')
