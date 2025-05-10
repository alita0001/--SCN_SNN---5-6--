#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 15:12:14 2021

@author: spiros
"""
import pickle
import pathlib
import keras
import numpy as np
import pandas as pd
import seaborn_image as isns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.transforms as mtransforms

from opt import get_data


def symmetrical_colormap(cmap_settings, new_name=None):
    """
    制作对称的颜色映射。
    
    该函数接收一个颜色映射并创建一个新的颜色映射，
    通过对自身进行对称折叠连接而成。
    https://stackoverflow.com/questions/28439251/symmetric-colormap-matplotlib

    参数:
    ----------
    cmap_settings : 元组
        包含两个元素的元组。第一个是颜色映射，第二个是
        离散化因子。例如: cmap_settings = ('Blues', None)
        提供整数而非None可以"离散化/分箱"颜色映射
    new_name : 字符串, 可选
        新颜色映射的名称。默认为None。

    返回:
    -------
    mymap : matplotlib.colors.LinearSegmentedColormap
        新创建的对称颜色映射
    """
    # 获取颜色映射
    cmap = plt.cm.get_cmap(*cmap_settings)
    if not new_name:
        new_name = "sym_"+cmap_settings[0]  # 示例: 'sym_Blues'
    # 定义颜色映射的精细度，128为精细值
    n = 128
    # 从颜色映射获取颜色列表
    # 获取标准颜色映射的 '右部分'
    colors_r = cmap(np.linspace(0, 1, n))
    # 获取第一个颜色列表并反转顺序 # "左部分"
    colors_l = colors_r[::-1]

    # 组合它们并构建新的颜色映射
    colors = np.vstack((colors_l, colors_r))
    mymap = mcolors.LinearSegmentedColormap.from_list(new_name, colors)

    return mymap


def my_style():
    """
    创建自定义绘图样式。

    返回:
    -------
    my_style : 字典
        包含matplotlib参数的字典。
    """
    # 字体大小
    fsize = 10
    # 定义绘图样式参数字典
    my_style = {
        # 使用LaTeX渲染所有文本
        "text.usetex": False,
        "font.family": "Arial",
        # "font.weight": "bold",  # 字体粗细
        # 在图中使用16pt字体，与文档中的16pt字体匹配
        "axes.labelsize": fsize,  # 轴标签大小
        "axes.titlesize": fsize,  # 标题大小
        "font.size": fsize,  # 字体大小
        "grid.color": "black",  # 网格颜色
        "grid.linewidth": 0.2,  # 网格线宽
        # 使图例/标签字体略小
        "legend.fontsize": fsize-2,  # 图例字体大小
        "xtick.labelsize": fsize,  # x轴刻度标签大小
        "ytick.labelsize": fsize,  # y轴刻度标签大小
        "axes.linewidth": 1.5,  # 轴线宽
        "lines.markersize": 4.0,  # 标记点大小
        "lines.linewidth": 1.0,  # 线宽
        "xtick.major.width": 1.2,  # x轴主刻度宽度
        "ytick.major.width": 1.2,  # y轴主刻度宽度
        "axes.edgecolor": "black",  # 轴边缘颜色
        # "axes.labelweight": "bold",  # 轴标签字体粗细
        # "axes.titleweight": "bold",  # 添加此行将标题字体设置为粗体
        "axes.spines.right": False,  # 隐藏右侧轴线
        "axes.spines.top": False,  # 隐藏顶部轴线
        "svg.fonttype": "none"  # SVG字体类型
    }

    return my_style


def set_size(width, fraction=1):
    """
    设置图形尺寸以避免在LaTeX中缩放。

    参数:
    ----------
    width: float
        文档文本宽度或列宽（点）
    fraction: float, 可选
        希望图形占据的宽度比例

    返回:
    -------
    fig_dim: tuple
        图形尺寸（英寸）
    """
    # 图形宽度（点）
    fig_width_pt = width * fraction
    # 从点转换为英寸
    inches_per_pt = 1 / 72.27
    # 黄金比例设置美观的图形高度
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2
    # 图形宽度（英寸）
    fig_width_in = fig_width_pt * inches_per_pt
    # 图形高度（英寸）
    fig_height_in = fig_width_in * golden_ratio
    fig_dim = (fig_width_in, fig_height_in)
    return fig_dim


def keep_models(df_all, names):
    """
    保留需要进一步测试的模型。

    参数:
    ----------
    df_all : DataFrame
        包含所有模型的数据框
    names : list
        要保留的模型名称列表

    返回:
    -------
    df : DataFrame
        只包含指定模型的数据框
    """
    df = pd.DataFrame()
    for n in names:
        df = pd.concat([df, df_all[(df_all['model'] == n)]])
    return df


def fix_names(df):
    """
    替换和修改数据框中的名称。

    参数:
    ----------
    df : DataFrame
        包含模型名称的数据框

    返回:
    -------
    df : DataFrame
        名称已修改的数据框
    """
    # 将长模型名称替换为简短易读的名称
    df = df.replace(['dend_ann_global_rfs'], 'dANN-GRF')
    df = df.replace(['dend_ann_local_rfs'], 'dANN-LRF')
    df = df.replace(['dend_ann_random'], 'dANN-R')
    df = df.replace(['dend_ann_all_to_all'], 'pdANN')
    df = df.replace(['vanilla_ann'], 'vANN')
    df = df.replace(['vanilla_ann_random'], 'vANN-R')
    df = df.replace(['vanilla_ann_local_rfs'], 'vANN-LRF')
    df = df.replace(['vanilla_ann_global_rfs'], 'vANN-GRF')
    df = df.replace(['sparse_ann'], 'sANN')
    df = df.replace(['sparse_ann_local_rfs'], 'sANN-LRF')
    df = df.replace(['sparse_ann_global_rfs'], 'sANN-GRF')
    df = df.replace(['sparse_ann_all_to_all'], 'psANN')
    df = df.replace(['vanilla_ann_dropout_0.2'], 'vANN-0.2')
    df = df.replace(['vanilla_ann_dropout_0.5'], 'vANN-0.5')
    df = df.replace(['vanilla_ann_dropout_0.8'], 'vANN-0.8')
    return df


def calculate_best_model(df):
    """
    计算传统DNN的统计数据。

    参数:
    ----------
    df : pandas.DataFrame
        包含模型评估数据的数据框

    返回:
    -------
    eval_metrics : numpy.ndarray
        包含不同模型配置评估指标的数组
    """
    # 定义不同的树突和胞体节点数量
    dendrites = [2**i for i in range(7)]  # 树突节点数量：1,2,4,8,16,32,64
    somata = [2**i for i in range(5, 10)]  # 胞体节点数量：32,64,128,256,512
    eval_metrics = []
    # 遍历所有树突和胞体节点的组合
    for d in dendrites:
        for s in somata:
            # 筛选特定配置的数据
            df_ = df[(df["num_dends"] == d) & (df["num_soma"] == s)]
            # 计算平均测试损失、准确率，并记录配置和参数
            eval_metrics.append(
                [np.mean(df_['test_loss']),
                 np.mean(df_['test_acc']), d, s,
                 np.mean(df_['trainable_params'])],
                )
    return np.array(eval_metrics)


def model_config(df, d, s, m):
    """
    提取模型配置。

    参数:
    ----------
    df : DataFrame
        包含模型数据的数据框
    d : int
        树突节点数量
    s : int
        胞体节点数量
    m : str
        模型名称

    返回:
    -------
    DataFrame
        匹配指定配置的数据框子集
    """
    # 返回特定树突数、胞体数和模型名称的数据
    return(df[(df['num_dends'] == d) & (df['num_soma'] == s) & (df['model'] == m)])


def keep_best_models_data(df, models_list):
    """
    保留特定模型数据。

    参数:
    ----------
    df : DataFrame
        包含所有模型数据的数据框
    models_list : dict
        要保留的模型列表，键为模型名称，值为配置参数

    返回:
    -------
    DataFrame
        包含最佳模型数据的数据框
    """
    df_ = pd.DataFrame()
    for model in models_list.keys():
        # 获取模型的树突和胞体配置
        d, s = int(models_list[model][0]), int(models_list[model][1])
        # 获取该配置的模型数据并添加到结果数据框
        df_ = pd.concat([df_, model_config(df, d, s, model)])
    return df_.reset_index()


def load_models(
        model_type, num_dends, num_soma, dirname,
        sigma, trained, n_trials=5,
    ):
    """
    加载模型。

    参数:
    ----------
    model_type : str
        模型类型
    num_dends : int
        树突节点数量
    num_soma : int
        胞体节点数量
    dirname : str
        模型保存目录
    sigma : float
        噪声参数sigma值
    trained : bool
        是否加载训练过的模型
    n_trials : int, 可选
        试验次数，默认为5

    返回:
    -------
    models_list : list
        加载的模型列表
    """
    models_list = []
    for t in range(1, n_trials+1):
        # 生成文件名后缀
        postfix = f"sigma_{sigma}_trial_{t}_dends_{num_dends}_soma_{num_soma}"
        if trained:
            # 训练过的模型文件路径
            fname_model = pathlib.Path(f"{dirname}/{model_type}/model_{postfix}.h5")
        else:
            # 未训练的模型文件路径
            fname_model = pathlib.Path(f"{dirname}/{model_type}/untrained_model_{postfix}.h5")
        # 加载模型并添加到列表
        models_list.append(keras.models.load_model(fname_model))
    return models_list


def load_best_models(model_list, names, dirname, sigma=0.0, trained=True):
    """
    加载最佳模型。

    参数:
    ----------
    model_list : dict
        模型配置列表
    names : list
        模型名称列表
    dirname : str
        模型保存目录
    sigma : float, 可选
        噪声参数，默认为0.0
    trained : bool, 可选
        是否加载训练过的模型，默认为True

    返回:
    -------
    models_all : dict
        加载的所有模型，按模型类型组织
    """
    models_all = {}
    for model, name in zip(model_list.keys(), names):
        # 获取模型的树突和胞体配置
        d, s = int(model_list[model][0]), int(model_list[model][1])
        # 加载指定配置的模型
        models_all[model] = load_models(
            name, d, s,
            dirname,
            sigma=sigma,
            trained=trained,
            )
    return models_all


def find_best_models(
        df, model_names, metric='accuracy', compare=True,
        baseline='vANN'
    ):
    """
    查找最佳模型。

    参数:
    ----------
    df : DataFrame
        包含模型评估数据的数据框
    model_names : list
        要考虑的模型名称列表
    metric : str, 可选
        评估指标，默认为'accuracy'，可选'loss'
    compare : bool, 可选
        是否与基准模型比较，默认为True
    baseline : str, 可选
        基准模型名称，默认为'vANN'

    返回:
    -------
    models_best : dict
        每种模型类型的最佳配置
    """
    # 除基准模型外的所有模型名称
    model_names_ = [n for n in model_names if n != baseline]
    if compare:
        # 计算基准模型的评估指标
        eval_metrics_base = calculate_best_model(df[df['model'] == baseline])
        models_best = {}
        if metric == 'accuracy':
            # 获取基准模型的最佳准确率
            best_acc = eval_metrics_base[np.argmax(eval_metrics_base[:, 1])][1]
            # 遍历每种模型类型寻找优于基准的配置
            for model_type in model_names_:
                eval_metrics = calculate_best_model(df[df['model'] == model_type])
                for metric_ in eval_metrics:
                    if metric_[1] > best_acc:
                        models_best[model_type] = metric_[2:]
                        break
            # 保存基准模型的最佳配置
            models_best[baseline] = eval_metrics_base[np.argmax(eval_metrics_base[:, 1])][2:]
            # 对于未找到优于基准配置的模型，保存其最佳配置
            for model_type in model_names_:
                if model_type not in models_best.keys():
                    eval_ = calculate_best_model(df[df['model'] == model_type])
                    models_best[model_type] = eval_[np.argmax(eval_[:, 1])][2:]
        elif metric == 'loss':
            # 获取基准模型的最佳损失
            best_loss = eval_metrics_base[np.argmin(eval_metrics_base[:, 0])][0]
            # 遍历每种模型类型寻找优于基准的配置
            for model_type in model_names_:
                eval_metrics = calculate_best_model(df[df['model'] == model_type])
                for metric_ in eval_metrics:
                    if metric_[0] < best_loss:
                        models_best[model_type] = metric_[2:]
                        break
            # 保存基准模型的最佳配置
            models_best[baseline] = eval_metrics_base[np.argmin(eval_metrics_base[:, 0])][2:]
            # 对于未找到优于基准配置的模型，保存其最佳配置
            for model_type in model_names_:
                if model_type not in models_best.keys():
                    eval_ = calculate_best_model(df[df['model'] == model_type])
                    models_best[model_type] = eval_[np.argmin(eval_[:, 0])][2:]
    else:
        # 不与基准比较，直接找每种模型的最佳配置
        models_best = {}
        for model_type in model_names:
            eval_ = calculate_best_model(df[df['model'] == model_type])
            if metric == 'loss':
                models_best[model_type] = eval_[np.argmin(eval_[:, 0])][2:]
            elif metric == 'accuracy':
                models_best[model_type] = eval_[np.argmax(eval_[:, 1])][2:]

    return models_best


def get_class_names(dataset, labels):
    """
    根据给定数据集获取类别名称。

    参数:
    ----------
    dataset : str
        数据集名称
    labels : array
        数据标签

    异常:
    ------
    ValueError
        当提供的数据集无效时引发

    返回:
    -------
    array
        类别名称数组
    """
    # 为不同数据集定义类别名称映射
    if dataset == "mnist":
        # MNIST数据集：手写数字0-9
        class_names = {
            0: "0", 1: "1", 2: "2", 3: "3", 4: "4",
            5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
        }
    elif dataset == "fmnist":
        # Fashion-MNIST数据集：10类服装和配饰
        class_names = {
            0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress",
            4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag",
            9: "Ankle boot",
        }
    elif dataset == "kmnist":
        # Kuzushiji-MNIST数据集：日本平假名字符
        class_names = {
            0: "お", 1: "き", 2: "す", 3: "つ", 4: "な",
            5: "は", 6: "ま", 7: "や", 8: "れ", 9: "を",
        }
    elif dataset == "emnist":
        # Extended-MNIST数据集：手写数字、大小写字母
        class_names = {
            0: "0", 1: "1", 2: "2", 3: "3", 4: "4",
            5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
            10: "A", 11: "B", 12: "C", 13: "D",
            14: "E", 15: "F", 16: "G", 17: "H",
            18: "I", 19: "J", 20: "K", 21: "L",
            22: "M", 23: "N", 24: "O", 25: "P",
            26: "Q", 27: "R", 28: "S", 29: "T",
            30: "U", 31: "V", 32: "W", 33: "X",
            34: "Y", 35: "Z", 36: "a", 37: "b",
            38: "d", 39: "e", 40: "f", 41: "g",
            42: "h", 43: "n", 44: "q", 45: "r",
            46: "t",
        }
    elif dataset == "cifar10":
        # CIFAR-10数据集：10类自然图像
        class_names = {
            0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
            5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck",
        }
    else:
        # 数据集无效时抛出异常
        raise ValueError("Invalid dataset. Supported datasets:"
                         "'mnist', 'fmnist', 'emnist', 'kmnist', 'cifar10'")

    # 定义获取类别名称的函数
    def get_class_name(label):
        return class_names.get(label, "Unknown")

    # 使用np.vectorize将函数应用到整个数组
    vectorized_get_class_name = np.vectorize(get_class_name)
    class_names_array = vectorized_get_class_name(labels)

    return class_names_array


def calculate_proj_scores(
        model_to_keep, dirname_figs, sigma=0.0,
        dim_method="tsne", datatype="fmnist",
        seq_tag="_sequential", learn_phase="trained",
        n_trials=5, rep=1
    ):
    """
    计算投影空间上的评分。

    参数:
    ----------
    model_to_keep : list
        要保留的模型名称
    dirname_figs : str
        图像保存目录
    sigma : float, 可选
        噪声参数，默认为0.0
    dim_method : str, 可选
        降维方法，默认为"tsne"
    datatype : str, 可选
        数据集类型，默认为"fmnist"
    seq_tag : str, 可选
        序列标签，默认为"_sequential"
    learn_phase : str, 可选
        学习阶段，默认为"trained"
    n_trials : int, 可选
        试验次数，默认为5
    rep : int, 可选
        重复次数，默认为1

    返回:
    -------
    df_scores : DataFrame
        评分数据框
    embeddings : dict
        嵌入结果
    """
    df_scores = pd.DataFrame()
    for model_name in model_to_keep:
        # 生成文件名后缀
        postfix = f"{dim_method}_{datatype}{seq_tag}_sigma_{sigma}_{learn_phase}_rep_{rep}"
        fname_store = pathlib.Path(f"{dirname_figs}/post_analysis_embeddings_{postfix}")
        # 加载嵌入结果
        with open(f'{fname_store}.pkl', 'rb') as file:
            results = pickle.load(file)
        scores = results['scores']
        embeddings = results['embeddings']
        # 遍历所有试验和层
        for t in range(1, n_trials + 1):
            for l in [2, 4, 5]:  # 对应树突层、胞体层和输出层
                scores_ = scores[model_name][f'trial_{t}'][f'layer_{l}']
                df_ = pd.DataFrame(index=[0])
                df_['model'] = model_name
                # 设置层名称
                if l == 2:
                    df_['layer'] = 'dendritic'  # 树突层
                elif l == 4:
                    df_['layer'] = 'somatic'    # 胞体层
                elif l == 5:
                    df_['layer'] = 'output'     # 输出层
                # 记录各种评分
                df_['silhouette'] = scores_[0]      # 轮廓系数
                df_['nh_score'] = scores_[1]        # 近邻保持评分
                df_['trustworthiness'] = scores_[2] # 可信度
                df_['trial'] = t  # 试验编号
                df_['sigma'] = sigma  # 噪声参数
                df_scores = pd.concat([df_scores, df_], ignore_index=True)
    return df_scores, embeddings


def draw_text_metrics(
        ax, xloc, yloc, metric, text,
        color='black', fontsize=9
    ):
    """
    在直方图上绘制文本指标。

    参数:
    ----------
    ax : matplotlib.axes.Axes
        绘图轴
    xloc : float
        x位置（轴坐标系）
    yloc : float
        y位置（轴坐标系）
    metric : float
        指标值
    text : str
        指标描述文本
    color : str, 可选
        文本颜色，默认为'black'
    fontsize : int, 可选
        字体大小，默认为9

    返回:
    -------
    None
    """
    # 在轴上添加文本，显示指标名称和值
    ax.text(
        x=xloc,
        y=yloc,
        transform=ax.transAxes,  # 使用轴坐标系
        s=f"{text}: {metric:.3f}",  # 格式化文本
        fontweight='demibold',  # 字体粗细
        fontsize=fontsize,  # 字体大小
        verticalalignment='top',  # 垂直对齐
        horizontalalignment='right',  # 水平对齐
        backgroundcolor='white',  # 背景色
        color=color  # 文本颜色
    )


def make_subplots(fig, fig_part, dataset="mnist", label="A"):
    """
    绘制子图。

    参数:
    ----------
    fig : matplotlib.figure.Figure
        matplotlib图形对象
    fig_part : matplotlib.figure.Figure
        图形的一部分
    dataset : str, 可选
        数据集名称，默认为"mnist"
    label : str, 可选
        子图标签，默认为"A"

    返回:
    -------
    None
    """
    # 加载数据集
    data, labels, img_height, img_width, channels = get_data(
        validation_split=0.1,
        dtype=dataset,
    )
    x_train = data['train']
    y_train = labels['train']
    x = x_train[0].reshape(img_width, img_height, channels).squeeze()

    # 根据类别数量创建镶嵌布局
    n_classes = len(set(y_train))

    if dataset == "emnist":
        # EMNIST有更多类别，需要更大的布局
        mosaic = np.arange((int(n_classes / 10) + 1)*10).reshape(5, 10).astype('str')
    else:
        # 其他数据集使用2x5的布局
        mosaic = np.arange(n_classes).reshape(2, 5).astype('str')

    # 创建子图镶嵌
    axd = fig_part.subplot_mosaic(
        mosaic,
        gridspec_kw={
            "wspace": 0.0,  # 水平间距
            "hspace": 0.0,  # 垂直间距
            "left": 0.0,    # 左边距
            },
        )

    # 在左上角添加物理标签
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    axd["0"].text(
        0.0, 1.0, label,
        transform=axd["0"].transAxes + trans,
        fontsize='large', va='bottom'
        )

    # 遍历所有子图
    for i, (labels, ax) in enumerate(axd.items()):
        # 移除x轴和y轴刻度
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        ax.grid(False)
        if i > n_classes - 1:
            continue
        # 获取该类别的第一个样本
        x = x_train[y_train == i][0].reshape(img_width, img_height, channels).squeeze()
        # 显示图像
        isns.imshow(
            x,
            gray=True if channels == 1 else False,  # 灰度模式
            cbar=False,  # 不显示颜色条
            square=True,  # 正方形图像
            ax=ax,  # 绘图轴
            )


def short_to_long_names(names):
    """
    导出模型的长名称。

    参数:
    ----------
    names : list
        模型短名称列表

    返回:
    -------
    long_names : list
        模型长名称列表
    """
    long_names = []
    # 将短名称映射到长名称
    for n in names:
        if n == "dANN-R":
            long_names.append("dend_ann_random")
        elif n == "dANN-LRF":
            long_names.append("dend_ann_local_rfs")
        elif n == "dANN-GRF":
            long_names.append("dend_ann_global_rfs")
        elif n == "pdANN":
            long_names.append("dend_ann_all_to_all")
        elif n == "sANN":
            long_names.append("sparse_ann")
        elif n == "sANN-LRF":
            long_names.append("sparse_ann_local_rfs")
        elif n == "sANN-GRF":
            long_names.append("sparse_ann_global_rfs")
        elif n == "psANN":
            long_names.append("sparse_ann_all_to_all")
        elif n == "vANN-R":
            long_names.append("vanilla_ann_random")
        elif n == "vANN-LRF":
            long_names.append("vanilla_ann_local_rfs")
        elif n == "vANN-GRF":
            long_names.append("vanilla_ann_global_rfs")
        elif n == "vANN":
            long_names.append("vanilla_ann")
    return long_names


def calc_eff_scores(df, form='acc'):
    """
    计算损失和准确率效率评分。

    参数:
    ----------
    df : pandas.DataFrame
        包含模型评估数据的数据框
    form : str, 可选
        评分形式，'acc'表示准确率，'loss'表示损失，默认为'acc'

    返回:
    -------
    df_out : DataFrame
        包含效率评分的数据框
    """
    df_out = pd.DataFrame()
    # 获取数据集名称
    datasets = df["data"].unique()
    
    for d in datasets:
        # 筛选特定数据集的数据
        dataset_df = df[df["data"] == d].copy()

        # 计算参数数量和训练轮数的因子
        factor_params = dataset_df["trainable_params"]  # 可训练参数数量
        factor_epochs = dataset_df["num_epochs_min"] + 1  # 训练轮数

        # 计算归一化因子
        f = np.log10(factor_params*factor_epochs)  # 取对数
        f /= f.min()  # 归一化

        if form == "acc":
            # 计算归一化准确率（考虑参数和轮数）
            dataset_df["normed_acc"] = (dataset_df["test_acc"] / 100) / f
            df_out = pd.concat([df_out, dataset_df], ignore_index=True)
        elif form == "loss":
            # 计算归一化损失（考虑参数和轮数）
            dataset_df["normed_loss"] = dataset_df["test_loss"] * f
            df_out = pd.concat([df_out, dataset_df], ignore_index=True)

    return df_out
