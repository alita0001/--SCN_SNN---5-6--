import numpy as np
# from tensorflow.keras.datasets import cifar10, cifar100
# from tensorflow.keras.utils import to_categorical
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, KMNIST, EMNIST
import os
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import load_iris, load_breast_cancer, fetch_openml
import time
from sklearn.preprocessing import LabelEncoder

# 设置随机种子，确保每次运行结果一致
np.random.seed(42)  # 使用与SCN_SNN.py相同的随机种子
torch.manual_seed(42)  # PyTorch随机数生成器的种子
torch.cuda.manual_seed_all(42)  # 如果使用GPU，设置所有GPU的种子

def prepare_kfold_cross_validation_uci(dataset_name, n_splits=5, normalize=False, random_state=5):
    """
    为UCI数据集准备K折交叉验证的数据
    
    参数:
    dataset_name: 数据集名称，如'iris', 'breast_cancer', 'ionosphere'等
    n_splits: 折叠数，默认为5
    normalize: 是否标准化特征，默认为True
    random_state: 随机种子，确保结果可复现
    
    返回:
    folds: 包含K个(train_data, train_label, test_data, test_label)元组的列表
    """
    # 根据数据集名称加载完整数据
    if dataset_name.lower() == 'iris':
        dataset = load_iris()
        X = dataset.data
        y = dataset.target
        num_classes = 3
    elif dataset_name.lower() == 'breast_cancer':
        dataset = load_breast_cancer()
        X = dataset.data
        y = dataset.target
        num_classes = 2
    elif dataset_name.lower() == 'ionosphere':
        # 从OpenML下载
        dataset = fetch_openml(name='ionosphere', version=1, as_frame=True)
        X = dataset.data.values
        y = (dataset.target == 'g').astype(int).values
        num_classes = 2
    elif dataset_name.lower() == 'liver' or dataset_name.lower() == 'liver_disorders':
        # 尝试加载肝脏疾病数据集
        try:
            dataset = fetch_openml(name='liver-disorders', version=1, as_frame=True)
            X = dataset.data.values
            
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(dataset.target.values)
            num_classes = len(np.unique(y))
        except:
            dataset = fetch_openml(data_id=8, as_frame=True)
            X = dataset.data.values
            
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(dataset.target.values)
            num_classes = len(np.unique(y))
    elif dataset_name.lower() == 'pima' or dataset_name.lower() == 'diabetes':
        # 加载糖尿病数据集
        dataset = fetch_openml(name='diabetes', version=1, as_frame=True)
        X = dataset.data.values
        y = (dataset.target == 'tested_positive').astype(int).values
        num_classes = 2
    else:
        raise ValueError(f"不支持的数据集名称: {dataset_name}")
    
    # 在分割前对数据进行随机打乱
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # 标准化特征（可选）
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # 创建K折交叉验证器
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # 准备K折数据
    folds = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # 转换为one-hot编码
        y_train_onehot = F.one_hot(torch.tensor(y_train, dtype=torch.long), num_classes).float().numpy()
        y_test_onehot = F.one_hot(torch.tensor(y_test, dtype=torch.long), num_classes).float().numpy()
        
        folds.append((X_train, y_train_onehot, X_test, y_test_onehot))
    
    return folds

def load_and_preprocess_cifar10(shuffle=True, grayscale=True, normalize=False):
    # from tensorflow.keras.datasets import cifar10
    # from tensorflow.keras.utils import to_categorical
    import torch
    import torchvision
    import torchvision.transforms as transforms
    import numpy as np
    import torch.nn.functional as F

    # 使用torchvision加载CIFAR-10数据集
    transform = transforms.Compose([transforms.ToTensor()])
    cifar_train = torchvision.datasets.CIFAR10(root=r"D:\研究生学习\常用数据集\CIFAR10", train=True, download=True, transform=transform)
    cifar_test = torchvision.datasets.CIFAR10(root=r"D:\研究生学习\常用数据集\CIFAR10", train=False, download=True, transform=transform)
    
    # 提取图像和标签
    train_images = []
    train_labels = []
    for image, label in cifar_train:
        if grayscale:
            # 将RGB转换为灰度 (使用权重: R*0.299 + G*0.587 + B*0.114)
            gray_image = image[0] * 0.299 + image[1] * 0.587 + image[2] * 0.114
            train_images.append(gray_image.reshape(-1).numpy())
        else:
            train_images.append(image.reshape(-1).numpy())
        train_labels.append(label)
    
    test_images = []
    test_labels = []
    for image, label in cifar_test:
        if grayscale:
            # 将RGB转换为灰度
            gray_image = image[0] * 0.299 + image[1] * 0.587 + image[2] * 0.114
            test_images.append(gray_image.reshape(-1).numpy())
        else:
            test_images.append(image.reshape(-1).numpy())
        test_labels.append(label)
    
    # 转换为numpy数组
    train_images = np.array(train_images)
    test_images = np.array(test_images)
    
    # 归一化到[0,1]范围（如果数据还不是归一化的）
    if normalize and not (image.max() <= 1.0):
        train_images = train_images / 255.0
        test_images = test_images / 255.0
    
    # 标签转换为one-hot编码
    train_labels = F.one_hot(torch.tensor(train_labels), 10).float().numpy()
    test_labels = F.one_hot(torch.tensor(test_labels), 10).float().numpy()
    
    # 随机打乱训练集
    if shuffle:
        indices = np.random.permutation(len(train_images))
        train_images = train_images[indices]
        train_labels = train_labels[indices]
    
    return (train_images, train_labels, test_images, test_labels)


def load_and_preprocess_cifar100(shuffle=True, grayscale=False, normalize=True):
    # 使用torchvision加载CIFAR-100数据集
    transform = transforms.ToTensor()
    train_dataset = CIFAR100(root=r"D:\研究生学习\常用数据集\CIFAR100", train=True, download=True, transform=transform)
    test_dataset = CIFAR100(root=r"D:\研究生学习\常用数据集\CIFAR100", train=False, download=True, transform=transform)
    
    # 提取数据和标签
    train_data = []
    train_labels = []
    for img, label in train_dataset:
        if grayscale:
            # 将RGB转换为灰度
            gray_img = img[0] * 0.299 + img[1] * 0.587 + img[2] * 0.114
            train_data.append(gray_img.flatten().numpy())
        else:
            train_data.append(img.flatten().numpy())
        train_labels.append(label)
    
    test_data = []
    test_labels = []
    for img, label in test_dataset:
        if grayscale:
            # 将RGB转换为灰度
            gray_img = img[0] * 0.299 + img[1] * 0.587 + img[2] * 0.114
            test_data.append(gray_img.flatten().numpy())
        else:
            test_data.append(img.flatten().numpy())
        test_labels.append(label)
    
    # 转换为numpy数组
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    
    # 归一化到[0,1]范围（如果数据还不是归一化的）
    if normalize and not (img.max() <= 1.0):
        train_data = train_data / 255.0
        test_data = test_data / 255.0
    
    # 将标签转换为one-hot编码
    train_labels = F.one_hot(torch.tensor(train_labels), 100).float().numpy()
    test_labels = F.one_hot(torch.tensor(test_labels), 100).float().numpy()

    # 随机打乱训练集
    if shuffle:
        indices = np.random.permutation(len(train_data))
        train_data = train_data[indices]
        train_labels = train_labels[indices]

    return (train_data, train_labels, test_data, test_labels)

def load_and_preprocess_dvs_gesture(time_steps=20, shuffle=True):
    # 导入所需的软件包
    from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
    import torch.nn.functional as F

    dir_root = r"D:\研究生学习\常用数据集\DVS128Gesture"
    train_dataset = DVS128Gesture(
        root=dir_root,
        train=True,
        data_type='frame',
        frames_number=time_steps,
        split_by='number'
    )
    test_dataset = DVS128Gesture(
        root=dir_root,
        train=False,
        data_type='frame',
        frames_number=time_steps,
        split_by='number'
    )

    train_data = []
    test_data = []
    train_labels = []
    test_labels = []

    for i in range(len(train_dataset)):
        frame, label = train_dataset[i]
        train_data.append(frame)
        train_labels.append(label)
        if i == 1:
            print(type(frame))
            print(type(label))
    for i in range(len(test_dataset)):
        frame, label = test_dataset[i]
        test_data.append(frame)
        test_labels.append(label)

    # 将标签转换为one-hot编码
    train_labels = F.one_hot(torch.tensor(train_labels), 11).float()
    test_labels = F.one_hot(torch.tensor(test_labels), 11).float()


    # 转换为numpy数组
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    # 随机打乱训练集（需要在重塑前进行）
    if shuffle:
        indices = np.random.permutation(len(train_data))
        train_data = train_data[indices]
        train_labels = train_labels[indices]

    train_data = train_data.reshape(train_data.shape[1], train_data.shape[0], -1)
    test_data = test_data.reshape(test_data.shape[1], test_data.shape[0], -1)

    print(type(train_labels))
    print(type(train_data))

    return (train_data, train_labels, test_data, test_labels)


def load_and_preprocess_N_MNIST(time_steps=10, shuffle=True):
    # 导入所需的软件包
    from spikingjelly.datasets.n_mnist import NMNIST
    # from tensorflow.keras.utils import to_categorical
    # import numpy as np

    dir_root = r"D:\研究生学习\常用数据集\NMNIST"
    train_dataset = NMNIST(
        root=dir_root,
        train=True,
        data_type='frame',
        frames_number=time_steps,
        split_by='number'
    )
    test_dataset = NMNIST(
        root=dir_root,
        train=False,
        data_type='frame',
        frames_number=time_steps,
        split_by='number'
    )

    train_data = []
    test_data = []
    train_labels = []
    test_labels = []

    for i in range(len(train_dataset)):
        frame, label = train_dataset[i]
        train_data.append(frame)
        train_labels.append(label)
    for i in range(len(test_dataset)):
        frame, label = test_dataset[i]
        test_data.append(frame)
        test_labels.append(label)

    # 将标签转换为one-hot编码
    train_labels = F.one_hot(torch.tensor(train_labels), 11).float().numpy()
    test_labels = F.one_hot(torch.tensor(test_labels), 11).float().numpy()

    # 转换为numpy数组
    train_data = np.array(train_data)
    test_data = np.array(test_data)

    # 随机打乱训练集（需要在重塑前进行）
    if shuffle:
        indices = np.random.permutation(len(train_data))
        train_data = train_data[indices]
        train_labels = train_labels[indices]

    # 转换为 [T, N, F] 的格式
    train_data = train_data.reshape(train_data.shape[1], train_data.shape[0], -1)
    test_data = test_data.reshape(test_data.shape[1], test_data.shape[0], -1)

    return (train_data, train_labels, test_data, test_labels)

def load_and_preprocess_mnist(shuffle=True):
    # from tensorflow.keras.datasets import mnist
    # from tensorflow.keras.utils import to_categorical
    import torch
    import torchvision
    import torchvision.transforms as transforms
    import numpy as np
    import torch.nn.functional as F

    # 使用torchvision加载MNIST数据集
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = torchvision.datasets.MNIST(root=r"D:\研究生学习\常用数据集\MNIST", train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST(root=r"D:\研究生学习\常用数据集\MNIST", train=False, download=True, transform=transform)
    
    # 提取图像和标签
    train_images = []
    train_labels = []
    for image, label in mnist_train:
        train_images.append(image.reshape(-1).numpy())
        train_labels.append(label)
    
    test_images = []
    test_labels = []
    for image, label in mnist_test:
        test_images.append(image.reshape(-1).numpy())
        test_labels.append(label)
    
    # 转换为numpy数组
    train_images = np.array(train_images)
    test_images = np.array(test_images)
    
    # 标签转换为one-hot编码
    train_labels = F.one_hot(torch.tensor(train_labels), 10).float().numpy()
    test_labels = F.one_hot(torch.tensor(test_labels), 10).float().numpy()
    
    # 随机打乱训练集
    if shuffle:
        indices = np.random.permutation(len(train_images))
        train_images = train_images[indices]
        train_labels = train_labels[indices]
    
    return (train_images, train_labels, test_images, test_labels)

#################################################
# 新增的数据集加载函数
#################################################

def load_and_preprocess_fashion_mnist(shuffle=True, normalize=False):
    """
    加载并预处理Fashion MNIST数据集
    """
    # 使用torchvision加载Fashion MNIST数据集
    transform = transforms.Compose([transforms.ToTensor()])
    fmnist_train = FashionMNIST(root=r"D:\研究生学习\常用数据集\FashionMNIST", train=True, download=True, transform=transform)
    fmnist_test = FashionMNIST(root=r"D:\研究生学习\常用数据集\FashionMNIST", train=False, download=True, transform=transform)
    
    # 提取图像和标签
    train_images = []
    train_labels = []
    for image, label in fmnist_train:
        train_images.append(image.reshape(-1).numpy())
        train_labels.append(label)
    
    test_images = []
    test_labels = []
    for image, label in fmnist_test:
        test_images.append(image.reshape(-1).numpy())
        test_labels.append(label)
    
    # 转换为numpy数组
    train_images = np.array(train_images)
    test_images = np.array(test_images)
    
    # 归一化到[0,1]范围（如果数据还不是归一化的）
    if normalize and not (train_images.max() <= 1.0):
        train_images = train_images / 255.0
        test_images = test_images / 255.0
    
    # 标签转换为one-hot编码
    train_labels = F.one_hot(torch.tensor(train_labels), 10).float().numpy()
    test_labels = F.one_hot(torch.tensor(test_labels), 10).float().numpy()
    
    # 随机打乱训练集
    if shuffle:
        indices = np.random.permutation(len(train_images))
        train_images = train_images[indices]
        train_labels = train_labels[indices]
    
    return (train_images, train_labels, test_images, test_labels)

def load_and_preprocess_kmnist(shuffle=True, normalize=False):
    """
    加载并预处理Kuzushiji-MNIST (KMNIST)数据集
    """
    # 使用torchvision加载KMNIST数据集
    transform = transforms.Compose([transforms.ToTensor()])
    kmnist_train = KMNIST(root=r"D:\研究生学习\常用数据集\KMNIST", train=True, download=True, transform=transform)
    kmnist_test = KMNIST(root=r"D:\研究生学习\常用数据集\KMNIST", train=False, download=True, transform=transform)
    
    # 提取图像和标签
    train_images = []
    train_labels = []
    for image, label in kmnist_train:
        train_images.append(image.reshape(-1).numpy())
        train_labels.append(label)
    
    test_images = []
    test_labels = []
    for image, label in kmnist_test:
        test_images.append(image.reshape(-1).numpy())
        test_labels.append(label)
    
    # 转换为numpy数组
    train_images = np.array(train_images)
    test_images = np.array(test_images)
    
    # 归一化到[0,1]范围（如果数据还不是归一化的）
    if normalize and not (train_images.max() <= 1.0):
        train_images = train_images / 255.0
        test_images = test_images / 255.0
    
    # 标签转换为one-hot编码
    train_labels = F.one_hot(torch.tensor(train_labels), 10).float().numpy()
    test_labels = F.one_hot(torch.tensor(test_labels), 10).float().numpy()
    
    # 随机打乱训练集
    if shuffle:
        indices = np.random.permutation(len(train_images))
        train_images = train_images[indices]
        train_labels = train_labels[indices]
    
    return (train_images, train_labels, test_images, test_labels)

def load_and_preprocess_emnist(split='balanced', shuffle=True, normalize=False):
    """
    加载并预处理EMNIST数据集
    
    参数:
    split: EMNIST数据集的子集，可选值有'byclass', 'bymerge', 'balanced', 'letters', 'digits', 'mnist'
    shuffle: 是否打乱训练集
    normalize: 是否归一化数据
    """
    # 使用torchvision加载EMNIST数据集
    transform = transforms.Compose([transforms.ToTensor()])
    emnist_train = EMNIST(root=r"D:\研究生学习\常用数据集\EMNIST", split=split, train=True, download=True, transform=transform)
    emnist_test = EMNIST(root=r"D:\研究生学习\常用数据集\EMNIST", split=split, train=False, download=True, transform=transform)
    
    # 获取类别数量
    num_classes = len(emnist_train.classes)
    
    # 提取图像和标签
    train_images = []
    train_labels = []
    for image, label in emnist_train:
        train_images.append(image.reshape(-1).numpy())
        train_labels.append(label)
    
    test_images = []
    test_labels = []
    for image, label in emnist_test:
        test_images.append(image.reshape(-1).numpy())
        test_labels.append(label)
    
    # 转换为numpy数组
    train_images = np.array(train_images)
    test_images = np.array(test_images)
    
    # 归一化到[0,1]范围（如果数据还不是归一化的）
    if normalize and not (train_images.max() <= 1.0):
        train_images = train_images / 255.0
        test_images = test_images / 255.0
    
    # 标签转换为one-hot编码
    train_labels = F.one_hot(torch.tensor(train_labels), num_classes).float().numpy()
    test_labels = F.one_hot(torch.tensor(test_labels), num_classes).float().numpy()
    
    # 随机打乱训练集
    if shuffle:
        indices = np.random.permutation(len(train_images))
        train_images = train_images[indices]
        train_labels = train_labels[indices]
    
    return (train_images, train_labels, test_images, test_labels)

def load_and_preprocess_iris(test_size=0.2, shuffle=True, normalize=False, cross_validation=False, n_splits=5, fold_idx=0):
    """
    加载并预处理Iris数据集
    
    参数:
    test_size: 测试集比例
    shuffle: 是否打乱数据
    normalize: 是否标准化特征
    cross_validation: 是否使用交叉验证
    n_splits: 交叉验证折数
    fold_idx: 当前使用的折索引(0到n_splits-1)
    """
    # 加载Iris数据集
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # 标准化特征（可选）
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    if cross_validation:
        # 使用K折交叉验证
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=42)
        
        # 获取所有折
        splits = list(kf.split(X))
        
        # 使用指定的折
        if fold_idx < 0 or fold_idx >= n_splits:
            raise ValueError(f"fold_idx必须在0到{n_splits-1}之间")
            
        train_indices, test_indices = splits[fold_idx]
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
    else:
        # 使用普通的训练/测试集划分
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=shuffle)
    
    # 转换为one-hot编码
    y_train_onehot = F.one_hot(torch.tensor(y_train, dtype=torch.long), 3).float().numpy()
    y_test_onehot = F.one_hot(torch.tensor(y_test, dtype=torch.long), 3).float().numpy()
    
    return (X_train, y_train_onehot, X_test, y_test_onehot)

def load_and_preprocess_breast_cancer(test_size=0.2, shuffle=True, normalize=False):
    """
    加载并预处理乳腺癌数据集
    """
    # 加载乳腺癌数据集
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target
    
    # 标准化特征（可选）
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=5, shuffle=shuffle)
    
    # 转换为one-hot编码
    y_train_onehot = F.one_hot(torch.tensor(y_train, dtype=torch.long), 2).float().numpy()
    y_test_onehot = F.one_hot(torch.tensor(y_test, dtype=torch.long), 2).float().numpy()
    
    return (X_train, y_train_onehot, X_test, y_test_onehot)

def load_and_preprocess_pima_diabetes(test_size=0.2, shuffle=True, normalize=False, data_path=None):
    """
    加载并预处理Pima Indians糖尿病数据集
    
    参数:
    data_path: 数据集路径，如果为None则从OpenML下载
    """
    if data_path:
        # 从本地加载
        df = pd.read_csv(data_path)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
    else:
        # 从OpenML下载
        diabetes = fetch_openml(name='diabetes', version=1, as_frame=True)
        X = diabetes.data.values
        y = (diabetes.target == 'tested_positive').astype(int).values
    
    # 标准化特征（可选）
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=shuffle)
    
    # 转换为one-hot编码
    y_train_onehot = F.one_hot(torch.tensor(y_train, dtype=torch.long), 2).float().numpy()
    y_test_onehot = F.one_hot(torch.tensor(y_test, dtype=torch.long), 2).float().numpy()
    
    return (X_train, y_train_onehot, X_test, y_test_onehot)

def load_and_preprocess_liver_disorders(test_size=0.2, shuffle=True, normalize=False, data_path=None):
    """
    加载并预处理肝脏疾病数据集
    
    参数:
    data_path: 数据集路径，如果为None则尝试从OpenML下载
    """
    if data_path:
        # 从本地加载
        try:
            df = pd.read_csv(data_path)
            print(f"成功从{data_path}加载肝脏疾病数据集")
            
            # 假设最后一列是标签
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
        except Exception as e:
            raise RuntimeError(f"加载本地数据集失败: {e}")
    else:
        # 尝试从OpenML下载
        try:
            print("正在从OpenML下载肝脏疾病数据集...")
            liver = fetch_openml(name='liver-disorders', version=1, as_frame=True)
            X = liver.data.values
            y = liver.target.astype(int).values
            print("成功从OpenML下载肝脏疾病数据集")
        except Exception as e:
            raise RuntimeError(f"从OpenML下载数据集失败: {e}")
    
    # 确保标签是从0开始的整数
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # 标准化特征（可选）
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=shuffle)
    
    # 确定类别数量
    num_classes = len(np.unique(y))
    print(f"肝脏数据集类别数量: {num_classes}, 类别值范围: {np.min(y)}~{np.max(y)}")
    
    # 转换为one-hot编码
    y_train_onehot = F.one_hot(torch.tensor(y_train, dtype=torch.long), num_classes).float().numpy()
    y_test_onehot = F.one_hot(torch.tensor(y_test, dtype=torch.long), num_classes).float().numpy()
    
    return (X_train, y_train_onehot, X_test, y_test_onehot)

def load_and_preprocess_ionosphere(test_size=0.2, shuffle=True, normalize=False, data_path=None):
    """
    加载并预处理电离层数据集
    
    参数:
    data_path: 数据集路径，如果为None则从OpenML下载
    """
    if data_path:
        # 从本地加载
        df = pd.read_csv(data_path)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
    else:
        # 从OpenML下载
        ionosphere = fetch_openml(name='ionosphere', version=1, as_frame=True)
        X = ionosphere.data.values
        y = (ionosphere.target == 'g').astype(int).values
    
    # 标准化特征（可选）
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=shuffle)
    
    # 转换为one-hot编码
    y_train_onehot = F.one_hot(torch.tensor(y_train, dtype=torch.long), 2).float().numpy()
    y_test_onehot = F.one_hot(torch.tensor(y_test, dtype=torch.long), 2).float().numpy()
    
    return (X_train, y_train_onehot, X_test, y_test_onehot)

def load_uci_dataset(dataset_name, test_size=0.2, shuffle=True, normalize=False, data_path=None):
    """
    通用函数，用于加载和预处理UCI数据集
    
    参数:
    dataset_name: UCI数据集名称
    data_path: 数据集路径，如果为None则从OpenML下载
    """
    if data_path:
        # 从本地加载
        df = pd.read_csv(data_path)
        # 假设最后一列是标签
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
    else:
        # 从OpenML下载
        dataset = fetch_openml(name=dataset_name, as_frame=True)
        X = dataset.data.values
        y = dataset.target.values
        
        # 如果目标是字符串类型，进行编码
        if isinstance(y[0], str):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)
    
    # 标准化特征（可选）
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=shuffle)
    
    # 确定类别数量
    num_classes = len(np.unique(y))
    
    # 转换为one-hot编码 - 确保使用长整型
    y_train_onehot = F.one_hot(torch.tensor(y_train.astype(int), dtype=torch.long), num_classes).float().numpy()
    y_test_onehot = F.one_hot(torch.tensor(y_test.astype(int), dtype=torch.long), num_classes).float().numpy()
    
    return (X_train, y_train_onehot, X_test, y_test_onehot)

def load_and_preprocess_echocardiogram(test_size=0.2, shuffle=True, normalize=False , data_path=None):
    """
    加载并预处理Echocardiogram数据集
    """
    if data_path:
        # 从本地加载
        df = pd.read_csv(data_path)
    else:
        # 从OpenML下载
        try:
            dataset = fetch_openml(name='echocardiogram', version=1, as_frame=True)
            df = dataset.data.copy()
            target = dataset.target
        except Exception as e:
            print(f"尝试加载echocardiogram失败: {e}")
            # 尝试使用数据集ID
            try:
                dataset = fetch_openml(data_id=989, as_frame=True)
                df = dataset.data.copy()
                target = dataset.target
            except Exception as e2:
                raise RuntimeError(f"加载echocardiogram数据集失败: {e2}")
    
    # 确保DataFrame中没有缺失值，用众数填充
    df = df.fillna(df.mode().iloc[0])
    
    # 处理所有列
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            # 将非数值列转换为类别编码
            df[col] = pd.Categorical(df[col]).codes
    
    # 提取特征和标签
    if data_path:
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
    else:
        X = df.values
        # 转换标签
        le = LabelEncoder()
        y = le.fit_transform(target.values)
    
    # 标准化特征（可选）
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=shuffle)
    
    # 确定类别数量
    num_classes = len(np.unique(y))
    
    # 转换为one-hot编码
    y_train_onehot = F.one_hot(torch.tensor(y_train, dtype=torch.long), num_classes).float().numpy()
    y_test_onehot = F.one_hot(torch.tensor(y_test, dtype=torch.long), num_classes).float().numpy()
    
    return (X_train, y_train_onehot, X_test, y_test_onehot)

def load_and_preprocess_mammogram(test_size=0.2, shuffle=True, normalize=False, data_path=None):
    """
    加载并预处理Mammogram数据集
    
    参数:
    test_size: 测试集比例
    shuffle: 是否打乱数据
    normalize: 是否标准化特征
    data_path: 数据集本地路径，如果为None则从OpenML下载
    """
    if data_path:
        # 从本地加载
        df = pd.read_csv(data_path)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
    else:
        # 从OpenML下载
        try:
            dataset = fetch_openml(name='mammographic-mass', version=1, as_frame=True)
            X = dataset.data.values
            
            # 确保标签是从0开始的整数
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(dataset.target.values)
        except Exception as e:
            print(f"尝试加载mammographic-mass失败: {e}")
            # 尝试使用数据集ID
            try:
                dataset = fetch_openml(data_id=1046, as_frame=True)
                X = dataset.data.values
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = le.fit_transform(dataset.target.values)
            except Exception as e2:
                raise RuntimeError(f"加载mammogram数据集失败: {e2}")
    
    # 标准化特征（可选）
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=shuffle)
    
    # 确定类别数量
    num_classes = len(np.unique(y))
    
    # 转换为one-hot编码
    y_train_onehot = F.one_hot(torch.tensor(y_train, dtype=torch.long), num_classes).float().numpy()
    y_test_onehot = F.one_hot(torch.tensor(y_test, dtype=torch.long), num_classes).float().numpy()
    
    return (X_train, y_train_onehot, X_test, y_test_onehot)

def load_and_preprocess_hepatitis(test_size=0.2, shuffle=True, normalize=False, data_path=None):
    """
    加载并预处理Hepatitis数据集
    """
    print(f'开始加载hepatitis数据集')
    if data_path:
        # 从本地加载
        df = pd.read_csv(data_path)
    else:
        # 从OpenML下载
        try:
            dataset = fetch_openml(name='hepatitis', version=1, as_frame=True)
            df = dataset.data.copy()
            target = dataset.target
        except Exception as e:
            print(f"尝试加载hepatitis失败: {e}")
            # 尝试使用数据集ID
            try:
                dataset = fetch_openml(data_id=55, as_frame=True)
                df = dataset.data.copy()
                target = dataset.target
            except Exception as e2:
                raise RuntimeError(f"加载hepatitis数据集失败: {e2}")
    
    # 确保DataFrame中没有缺失值，用众数填充
    df = df.fillna(df.mode().iloc[0])
    
    # 处理所有列
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            # 将非数值列转换为类别编码
            df[col] = pd.Categorical(df[col]).codes
    
    # 提取特征和标签
    if data_path:
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
    else:
        X = df.values
        # 转换标签
        le = LabelEncoder()
        y = le.fit_transform(target.values)
    
    # 对每个特征维度单独进行归一化（可选）
    if normalize:
        # 对每一列单独进行Min-Max归一化处理到[0,1]区间
        for i in range(X.shape[1]):
            col_min = np.min(X[:, i])
            col_max = np.max(X[:, i])
            print(f'第{i}列的最小值为{col_min},最大值为{col_max}')
            if col_max > col_min:  # 避免除以零
                X[:, i] = (X[:, i] - col_min) / (col_max - col_min)
            else:
                X[:, i] = 0  # 如果所有值相同，则设为0
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=shuffle)
    
    # 确定类别数量
    num_classes = len(np.unique(y))
    
    # 转换为one-hot编码
    y_train_onehot = F.one_hot(torch.tensor(y_train, dtype=torch.long), num_classes).float().numpy()
    y_test_onehot = F.one_hot(torch.tensor(y_test, dtype=torch.long), num_classes).float().numpy()
    
    return (X_train, y_train_onehot, X_test, y_test_onehot)

def load_and_preprocess_wine(test_size=0.2, shuffle=True, normalize=False, data_path=None):
    """
    加载并预处理Wine数据集
    
    参数:
    test_size: 测试集比例
    shuffle: 是否打乱数据
    normalize: 是否标准化特征
    data_path: 数据集本地路径，如果为None则使用scikit-learn内置的数据集
    """
    if data_path:
        # 从本地加载
        df = pd.read_csv(data_path)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
    else:
        # 使用scikit-learn内置的Wine数据集
        from sklearn.datasets import load_wine
        wine = load_wine()
        X = wine.data
        y = wine.target
    
    # 标准化特征（可选）
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=shuffle)
    
    # 确定类别数量
    num_classes = len(np.unique(y))
    
    # 转换为one-hot编码
    y_train_onehot = F.one_hot(torch.tensor(y_train, dtype=torch.long), num_classes).float().numpy()
    y_test_onehot = F.one_hot(torch.tensor(y_test, dtype=torch.long), num_classes).float().numpy()
    
    return (X_train, y_train_onehot, X_test, y_test_onehot)

def load_and_preprocess_acoustic_emission(test_size=0.2, shuffle=True, normalize=False, data_path=None):
    """
    加载并预处理Acoustic Emission数据集
    
    参数:
    test_size: 测试集比例
    shuffle: 是否打乱数据
    normalize: 是否标准化特征
    data_path: 数据集本地路径，如果为None则从OpenML下载
    """
    if data_path:
        # 从本地加载
        df = pd.read_csv(data_path)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
    else:
        # 从OpenML下载
        try:
            dataset = fetch_openml(name='acoustic', version=1, as_frame=True)
            X = dataset.data.values
            
            # 确保标签是从0开始的整数
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(dataset.target.values)
        except Exception as e:
            print(f"尝试加载acoustic emission数据集失败: {e}")
            # 尝试使用更通用的名称
            try:
                dataset = fetch_openml(name='acoustic-emission', version=1, as_frame=True)
                X = dataset.data.values
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = le.fit_transform(dataset.target.values)
            except Exception as e2:
                print(f"尝试加载acoustic-emission数据集失败: {e2}")
                # 最后尝试使用大规模声学数据集
                try:
                    dataset = fetch_openml(name='phoneme', version=1, as_frame=True)
                    X = dataset.data.values
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    y = le.fit_transform(dataset.target.values)
                except Exception as e3:
                    raise RuntimeError(f"加载acoustic emission相关数据集失败: {e3}")
    
    # 标准化特征（可选）
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=shuffle)
    
    # 确定类别数量
    num_classes = len(np.unique(y))
    
    # 转换为one-hot编码
    y_train_onehot = F.one_hot(torch.tensor(y_train, dtype=torch.long), num_classes).float().numpy()
    y_test_onehot = F.one_hot(torch.tensor(y_test, dtype=torch.long), num_classes).float().numpy()
    
    return (X_train, y_train_onehot, X_test, y_test_onehot)

def load_dataset(dataset_name, **kwargs):
    """
    通用数据集加载函数
    
    参数:
    dataset_name: 数据集名称，支持以下数据集：
        - 'mnist', 'cifar10', 'cifar100'
        - 'fmnist', 'kmnist', 'emnist'
        - 'iris', 'breast_cancer', 'diabetes', 'liver', 'ionosphere'
        - 'dvs', 'nmnist'
    kwargs: 传递给特定数据集加载函数的参数
    
    返回:
    (train_data, train_labels, test_data, test_labels): 加载并预处理好的数据集
    """
    dataset_name = dataset_name.lower().strip()
    
    # 图像数据集
    if dataset_name == 'mnist':
        return load_and_preprocess_mnist(**kwargs)
    elif dataset_name == 'cifar10':
        return load_and_preprocess_cifar10(**kwargs)
    elif dataset_name == 'cifar100':
        return load_and_preprocess_cifar100(**kwargs)
    elif dataset_name == 'fmnist' or dataset_name == 'fashion_mnist':
        return load_and_preprocess_fashion_mnist(**kwargs)
    elif dataset_name == 'kmnist':
        return load_and_preprocess_kmnist(**kwargs)
    elif dataset_name == 'emnist':
        split = kwargs.pop('split', 'balanced')
        return load_and_preprocess_emnist(split=split, **kwargs)
    
    # 事件数据集
    elif dataset_name == 'dvs' or dataset_name == 'dvs_gesture':
        return load_and_preprocess_dvs_gesture(**kwargs)
    elif dataset_name == 'nmnist':
        return load_and_preprocess_N_MNIST(**kwargs)
        
    # UCI数据集
    elif dataset_name == 'iris':
        return load_and_preprocess_iris(**kwargs)
    elif dataset_name == 'breast_cancer':
        return load_and_preprocess_breast_cancer(**kwargs)
    elif dataset_name == 'diabetes' or dataset_name == 'pima':
        return load_and_preprocess_pima_diabetes(**kwargs)
    elif dataset_name == 'liver' or dataset_name == 'liver_disorders':
        return load_and_preprocess_liver_disorders(**kwargs)
    elif dataset_name == 'ionosphere':
        return load_and_preprocess_ionosphere(**kwargs)
    elif dataset_name.lower() == 'echocardiogram':
        return load_and_preprocess_echocardiogram(**kwargs)
    elif dataset_name.lower() == 'mammogram' or dataset_name.lower() == 'mammographic-mass':
        return load_and_preprocess_mammogram(**kwargs)
    elif dataset_name.lower() == 'hepatitis':
        return load_and_preprocess_hepatitis(**kwargs)
    elif dataset_name.lower() == 'wine':
        return load_and_preprocess_wine(**kwargs)
    elif dataset_name.lower() == 'acoustic' or dataset_name.lower() == 'acoustic-emission':
        return load_and_preprocess_acoustic_emission(**kwargs)
    else:
        # 尝试作为UCI数据集加载
        try:
            print(f"尝试作为OpenML/UCI数据集加载 '{dataset_name}'...")
            return load_uci_dataset(dataset_name, **kwargs)
        except Exception as e:
            raise ValueError(f"不支持的数据集: {dataset_name}, 错误信息: {e}")

if __name__ == "__main__":
    # 测试所有数据集加载
    
    # 图像数据集列表
    image_datasets = ['mnist', 'cifar10', 'cifar100', 'fmnist', 'kmnist', 'emnist']
    # 事件数据集列表
    # event_datasets = ['dvs', 'nmnist']
    event_datasets = ['dvs']  
    # UCI数据集列表
    uci_datasets = ['iris', 'breast_cancer', 'diabetes', 'liver', 'ionosphere', 'echocardiogram', 'mammogram', 'hepatitis', 'wine', 'acoustic-emission']
    
    # 创建测试结果表
    results = []
    
    print("\n" + "="*60)
    print("测试所有支持的数据集加载功能")
    print("="*60)
    
    # 测试图像数据集
    print("\n图像数据集测试:")
    print("-"*40)
    for dataset in image_datasets:
        try:
            start_time = time.time()
            
            if dataset == 'emnist':  # EMNIST需要指定split参数
                train_data, train_labels, test_data, test_labels = load_dataset(dataset, split='balanced')
            else:
                # 对于cifar系列，使用grayscale=True简化数据
                if dataset.startswith('cifar'):
                    train_data, train_labels, test_data, test_labels = load_dataset(dataset, grayscale=True)
                else:
                    train_data, train_labels, test_data, test_labels = load_dataset(dataset)
            
            load_time = time.time() - start_time
            
            results.append({
                'dataset': dataset,
                'status': '成功',
                'train_shape': train_data.shape,
                'train_labels_shape': train_labels.shape,
                'test_shape': test_data.shape,
                'test_labels_shape': test_labels.shape,
                'load_time': f"{load_time:.2f}秒"
            })
            
            print(f"{dataset:10} - 加载成功! 训练数据: {train_data.shape}, 标签: {train_labels.shape}")
        except Exception as e:
            results.append({
                'dataset': dataset,
                'status': '失败',
                'error': str(e),
                'load_time': 'N/A'
            })
            print(f"{dataset:10} - 加载失败: {e}")
    
    # 测试事件数据集
    print("\n事件数据集测试:")
    print("-"*40)
    for dataset in event_datasets:
        try:
            start_time = time.time()
            
            # 事件数据集可能需要指定time_steps参数
            train_data, train_labels, test_data, test_labels = load_dataset(dataset, time_steps=10)
            
            load_time = time.time() - start_time
            
            results.append({
                'dataset': dataset,
                'status': '成功',
                'train_shape': train_data.shape,
                'train_labels_shape': train_labels.shape,
                'test_shape': test_data.shape,
                'test_labels_shape': test_labels.shape,
                'load_time': f"{load_time:.2f}秒"
            })
            
            print(f"{dataset:10} - 加载成功! 训练数据: {train_data.shape}, 标签: {train_labels.shape}")
        except Exception as e:
            results.append({
                'dataset': dataset,
                'status': '失败',
                'error': str(e),
                'load_time': 'N/A'
            })
            print(f"{dataset:10} - 加载失败: {e}")
    
    # 测试UCI数据集
    print("\nUCI数据集测试:")
    print("-"*40)
    for dataset in uci_datasets:
        try:
            start_time = time.time()
            train_data, train_labels, test_data, test_labels = load_dataset(dataset)
            elapsed_time = time.time() - start_time
            
            # 检查数据是否有效
            valid_data = True
            error_msg = None
            
            # 添加到结果
            results.append({
                'dataset': dataset,  # 使用'dataset'作为键名
                'status': '成功',
                'train_shape': train_data.shape,
                'test_shape': test_data.shape,
                'load_time': f"{elapsed_time:.2f}秒"
            })
            
            # 打印结果
            print(f"数据集: {dataset:<15} | 状态: {'成功':<8} | 训练数据: {str(train_data.shape):<20} | 测试数据: {str(test_data.shape):<20} | 加载时间: {elapsed_time:.2f}秒")
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"数据集: {dataset:<15} | 状态: {'失败':<8} | 错误: {str(e)[:100]}")
            results.append({
                'dataset': dataset,  # 使用'dataset'作为键名
                'status': '失败',
                'train_shape': None,
                'test_shape': None,
                'load_time': 'N/A',
                'error': str(e)[:100]
            })
    
    # 打印结果汇总
    print("\n" + "="*80)
    print("数据集加载测试结果汇总")
    print("="*80)
    print(f"{'数据集名称':15} | {'状态':8} | {'训练数据形状':22} | {'测试数据形状':22} | {'加载时间'}")
    print("-"*80)
    for result in results:
        if result['status'] == '成功':
            print(f"{result['dataset']:15} | {result['status']:8} | {str(result['train_shape']):22} | {str(result['test_shape']):22} | {result['load_time']}")
        else:
            print(f"{result['dataset']:15} | {result['status']:8} | {'N/A':22} | {'N/A':22} | {'N/A':5}")
    
    # 统计成功加载的数据集数量
    success_count = sum(1 for r in results if r['status'] == '成功')
    total_count = len(results)
    print(f"\n成功加载的数据集数量: {success_count}/{total_count}")
    
    # 如果有失败的数据集，打印错误详情
    if success_count < total_count:
        print("\n失败的数据集详情:")
        for result in results:
            if result['status'] == '失败':
                print(f"{result['dataset']}: {result.get('error', 'Unknown error')}")
