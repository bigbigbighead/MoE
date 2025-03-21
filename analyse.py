import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import time
import csv
import os
import matplotlib.pyplot as plt
import utils.train_and_test
from utils.data_loading import import_data, get_loaders
from utils.model_loading import load_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from infer import detailed_inference_test, load_model_parameters, read_data, get_set_info


def plot_confusion_matrix(cm, class_names=None, save_path="confusion_matrix.png", top_n=10):
    """
    绘制并保存混淆矩阵的热图

    Parameters
    ----------
    cm : np.ndarray
        混淆矩阵（大小为 [num_classes, num_classes]）
    class_names : list, optional
        类别名称列表（长度应等于 num_classes）
    save_path : str, optional
        保存热图的路径
    top_n : int, optional
        显示前 N 个分类错误最多的类别
    """
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", square=True)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix Heatmap")

    if class_names:
        plt.xticks(ticks=np.arange(len(class_names)), labels=class_names, rotation=90)
        plt.yticks(ticks=np.arange(len(class_names)), labels=class_names, rotation=0)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Confusion matrix heatmap saved to {save_path}")
    plt.show()


def get_misclass(class_s, class_d, cm):
    """
    返回本来是class_s，误分类成class_d的数量和比例
    """
    s_totals = cm[class_s].sum(axis=0)
    proportion = cm[class_s][class_d] / s_totals
    return cm[class_s][class_d], proportion


def analyze_most_confused_classes(loader_test, train_count, cm, class_names=None, top_n=10):
    """
    找出分类准确率最低的类别，并分析其误分类情况

    Parameters
    ----------
    cm : np.ndarray
        混淆矩阵
    class_names : list, optional
        类别名称列表
    top_n : int, optional
        显示前 N 个分类错误最多的类别
    """
    num_classes = cm.shape[0]

    # 计算每个类别的样本总数
    class_totals = cm.sum(axis=1)
    correct_predictions = np.diag(cm)  # 提取对角线元素（正确分类的数量）
    class_accuracy = correct_predictions / class_totals  # 分类准确率

    # 找到准确率最低的类别
    sorted_indices = np.argsort(class_accuracy)[:top_n]  # 取前 N 个分类最差的类别
    class_feature_matrices = extract_class_features(loader_test, num_classes)
    print("\n=== Top Misclassified Classes ===")
    for idx in sorted_indices:
        class_name = class_names[idx] if class_names else f"Class {idx}"
        print(
            f"{class_name}: Train count = {train_count[idx]}\tTest count = {class_totals[idx]}\tAccuracy = {class_accuracy[idx] * 100:.2f}%")
        # 找到误分类最多的目标类别
        misclassified = cm[idx].copy()
        misclassified[idx] = 0  # 去掉正确分类的部分
        most_confused_idx = np.argsort(misclassified)[-10:]  # 取误分类最多的前 n 个类

        print(f"\tCommon misclassified labels:")
        for mis_idx in reversed(most_confused_idx):
            misclass_name = class_names[mis_idx] if class_names else f"Class {mis_idx}"
            mis_num, proportion = get_misclass(mis_idx, idx, cm)
            mean_s = class_feature_matrices[idx].mean(axis=0).flatten()
            mean_d = class_feature_matrices[mis_idx].mean(axis=0).flatten()
            similarity = cosine_similarity([mean_s], [mean_d])[0][0]
            print(
                f"\t-{misclass_name}: {cm[idx, mis_idx]} samples\t{(cm[idx, mis_idx] / class_totals[idx] * 100):.2f}%")
            print(
                f"\t\ttrue class {misclass_name} misclassified to {class_name}: {mis_num}, {proportion * 100:.2f}%")
            print(f"\t\tcosine similarity is {similarity}")
        print("")


def analyze_confusion_matrix(cm, filepath, file_name="confusion_matrix.csv"):
    print("analyzing confusion matrix")


def plot_classwise_accuracy(cm, class_names=None, save_path="classwise_accuracy.png"):
    """
    绘制每个类别的分类准确率柱状图

    Parameters
    ----------
    cm : np.ndarray
        混淆矩阵
    class_names : list, optional
        类别名称列表
    save_path : str, optional
        保存图像的路径
    """
    correct_predictions = np.diag(cm)
    class_totals = cm.sum(axis=1)
    class_accuracy = correct_predictions / class_totals

    # 按类别排序
    sorted_indices = np.argsort(class_accuracy)
    sorted_accuracy = class_accuracy[sorted_indices]
    sorted_class_names = [class_names[i] if class_names else f"Class {i}" for i in sorted_indices]

    # 绘制柱状图
    plt.figure(figsize=(12, 6))
    plt.bar(sorted_class_names, sorted_accuracy * 100, color="blue")
    plt.xlabel("Class ID")
    plt.ylabel("Accuracy (%)")
    plt.xticks(rotation=90, fontsize=8)  # 旋转 x 轴标签以避免重叠
    plt.title("Class-wise Accuracy (Sorted)")
    plt.tight_layout()

    # 保存和显示
    plt.savefig(save_path)
    print(f"Class-wise accuracy plot saved to {save_path}")
    plt.show()


def get_infer_info(xp, result_file="test_inference_result.csv"):
    file_path = os.path.join(xp, result_file)
    test_inference_result = []

    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            class_id, train_set_size, test_set_size, correct_count, accuracy = row
            test_inference_result.append((class_id, train_set_size, test_set_size, correct_count, accuracy))
            # test_inference_result.append(row)
    return test_inference_result


def extract_class_features(data_loader, num_classes, device="cuda"):
    """
    遍历 DataLoader，并按类别存储

    Parameters
    ----------
    data_loader : DataLoader
        PyTorch 数据加载器
    num_classes : int
        类别总数
    device : str, optional
        计算设备（默认 'cuda'）

    Returns
    -------
    class_feature_matrices : list of np.ndarray
        每个类别的特征矩阵 (num_samples, 20, 2)
    """
    class_features = {i: [] for i in range(num_classes)}  # 用字典存储每个类别的特征

    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        for i, label in enumerate(labels):
            class_features[label.item()].append(inputs[i].cpu().numpy())  # 存入对应类别

    # 将列表转换为 NumPy 数组
    class_feature_matrices = [np.array(class_features[i]) for i in range(num_classes)]
    return class_feature_matrices


def compute_class_similarity(class_feature_matrices, xp_dir, save_path="class_similarity.png"):
    """
    计算类别特征的余弦相似度，并绘制热图

    Parameters
    ----------
    class_feature_matrices : list of np.ndarray
        每个类别的特征矩阵 (list of (num_samples, 20, 2) arrays)
    save_path : str, optional
        保存热图的名称
    """
    file_path = os.path.join(xp_dir, save_path)
    num_classes = len(class_feature_matrices)
    # 计算每个类别的平均特征向量 (20 × 2) → 展平成 (1 × 40)
    avg_feature_vectors = np.array([features.mean(axis=0).flatten() for features in class_feature_matrices])

    # 计算类别之间的余弦相似度
    similarity_matrix = cosine_similarity(avg_feature_vectors)

    # 绘制余弦相似度热图
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, annot=False, cmap="coolwarm", fmt=".2f")
    plt.xlabel("Class ID")
    plt.ylabel("Class ID")
    plt.title("Class Feature Similarity (Cosine)")
    plt.tight_layout()

    plt.savefig(save_path)
    print(f"Feature similarity heatmap saved to {save_path}")
    plt.show()

    # 计算类别相似度并绘制热图
    compute_class_similarity(class_feature_matrices)


def compute_two_class_similarity(class_1, class_2):
    """
    计算两个类别的平均特征向量之间的余弦相似度

    Parameters
    ----------
    class_1 : np.ndarray
        类别 1 的样本特征矩阵 (num_samples, 20, 2)
    class_2 : np.ndarray
        类别 2 的样本特征矩阵 (num_samples, 20, 2)

    Returns
    -------
    similarity : float
        余弦相似度（范围：-1 到 1）
    """
    mean_1 = class_1.mean(axis=0).flatten()  # 计算类别 1 的均值并展平为 (1 × 40)
    mean_2 = class_2.mean(axis=0).flatten()  # 计算类别 2 的均值并展平为 (1 × 40)

    similarity = cosine_similarity([mean_1], [mean_2])[0][0]  # 计算余弦相似度
    return similarity


if __name__ == '__main__':
    train_size = 3477816
    validation_size = 869454
    test_size = 4830012
    start_time = time.time()
    # 加载模型和数据集,获得混淆矩阵
    model, loader_train, loader_test, xp_dir = load_model_parameters()
    train_count = get_set_info(loader_train)
    endtime1 = time.time()
    print(f"Time passed:{endtime1 - start_time}")

    # # 保存混淆矩阵
    # np.savetxt(filepath + file_name, cm, delimiter=',', fmt="%d")
    # # 读取混淆矩阵
    # cm = np.loadtxt(filepath + file_name, delimiter=',', dtype=int)
    # xp_dir = "./results/AppClassNet/top200/lexnet/1/"
    cm_file = "confusion_matrix.csv"
    cm = np.loadtxt(xp_dir + cm_file, delimiter=',', dtype=int)
    cm_figure = "full_confusion_matrix.png"
    num_classes = len(cm[0])
    # 画出混淆矩阵heatmap
    # plot_confusion_matrix(cm, save_path=xp_dir + cm_figure)

    # 调用混淆类别分析函数
    class_names = [f"Class {i}" for i in range(num_classes)]  # 生成类别名称
    analyze_most_confused_classes(loader_test, train_count, cm, class_names, top_n=10)
    test_inference_result = get_infer_info(xp_dir, result_file="test_inference_result.csv")
    endtime2 = time.time()
    print(f"Time passed:{endtime2 - start_time}")

    # 计算余弦相似度
