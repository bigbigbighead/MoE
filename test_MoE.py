import glob
import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
import csv
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics.pairwise import cosine_similarity

from models.MoE import MoEResNet18


def load_data(split, dataset_path):
    """加载数据集"""
    x = np.load(f"{dataset_path}/{split}_x.npy", mmap_mode='r')
    y = np.load(f"{dataset_path}/{split}_y.npy")

    print(f"{split} data shape: {x.shape}")

    # 根据输入数据的实际形状调整
    if len(x.shape) == 4:  # 如果已经是4D张量 [batch, channels, height, width]
        x = torch.tensor(x, dtype=torch.float32)
    elif len(x.shape) == 3:  # 如果是3D张量 [batch, height, width]
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)  # 添加通道维度
    else:
        # 如果是2D张量 [batch, features]，需要重塑为适合CNN的形状
        x = torch.tensor(x, dtype=torch.float32).reshape(-1, 1, 1, x.shape[1])

    y = torch.tensor(y, dtype=torch.long)
    return x, y


def load_model_from_checkpoint(checkpoint_path=None, num_classes=None, num_experts=None, device=None,
                               results_path=None):
    """从检查点加载MoE模型"""
    if results_path is None:
        results_path = "./results/AppClassNet/top200/MoE/2"  # 默认结果路径

    # 如果没有指定检查点路径，自动寻找最佳模型
    if checkpoint_path is None:
        # 首先寻找最佳验证准确率的检查点
        best_model_files = glob.glob(f"{results_path}/param/best_model_*.pth")
        if best_model_files:
            # 提取准确率并找到最高的
            accuracies = [float(f.split('_')[-1].split('.pth')[0]) for f in best_model_files]
            best_idx = accuracies.index(max(accuracies))
            checkpoint_path = best_model_files[best_idx]
            print(f"自动选择最佳模型检查点: {checkpoint_path} (准确率: {accuracies[best_idx]}%)")
        else:
            # 如果没有best_model检查点，寻找最新的epoch检查点
            checkpoint_files = glob.glob(f"{results_path}/param/model_epoch_*.pth")
            if checkpoint_files:
                # 提取轮次数并找到最大的
                epochs = [int(f.split('_')[-1].split('.')[0]) for f in checkpoint_files]
                max_epoch = max(epochs)
                checkpoint_path = f"{results_path}/param/model_epoch_{max_epoch}.pth"
                print(f"自动选择最新轮次检查点: {checkpoint_path} (轮次: {max_epoch})")
            else:
                raise FileNotFoundError(f"在 {results_path}/param 中未找到可用的检查点")

    # 确保检查点存在
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")

    print(f"加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 获取输入形状信息，从检查点中提取或使用默认值
    input_channels = 1
    input_height = 1
    input_width = 1024

    # 从检查点中提取模型配置
    if isinstance(checkpoint, dict):
        # 如果检查点包含专门的模型配置
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            input_channels = config.get('input_channels', input_channels)
            input_height = config.get('input_height', input_height)
            input_width = config.get('input_width', input_width)
        # 使用训练过程中可能记录的其他字段
        elif 'diversity_weight' in checkpoint:
            # 检查是否有关于输入形状的信息
            if 'input_channels' in checkpoint:
                input_channels = checkpoint.get('input_channels', input_channels)
                input_height = checkpoint.get('input_height', input_height)
                input_width = checkpoint.get('input_width', input_width)

        # 如果检查点包含专家数量信息，使用它
        if num_experts is None and 'num_experts' in checkpoint:
            num_experts = checkpoint['num_experts']

    # 确保参数有合理的默认值
    if num_classes is None:
        num_classes = 200  # AppClassNet默认类别数
    if num_experts is None:
        num_experts = 3  # 默认专家数

    # 创建模型实例
    model = MoEResNet18(
        num_classes=num_classes,
        num_experts=num_experts,
        routing_type='softmax',
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width
    )

    # 加载模型参数
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def evaluate_model(model, data_loader, device, output_dir, collect_features=False):
    """评估模型并返回详细的评估结果"""
    model.eval()
    all_preds = []
    all_labels = []
    all_routing_weights = []
    all_expert_outputs = []
    all_features = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 获取模型的完整输出
            if collect_features:
                # 如果需要提取特征，我们需要手动调用backbone获取特征
                feats = model.backbone(inputs)
                all_features.append(feats.cpu().numpy())
                combined_logits, routing_weights, stacked_experts, _ = model(inputs)
            else:
                combined_logits, routing_weights, stacked_experts, _ = model(inputs)

            # 收集预测结果和真实标签
            _, preds = torch.max(combined_logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 收集路由权重和专家输出
            all_routing_weights.append(routing_weights.cpu().numpy())
            all_expert_outputs.append(stacked_experts.cpu().numpy())

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 保存混淆矩阵
    np.savetxt(os.path.join(output_dir, "confusion_matrix.csv"), cm, delimiter=',', fmt="%d")

    # 转换为NumPy数组
    all_routing_weights = np.vstack(all_routing_weights)
    all_expert_outputs = np.vstack(all_expert_outputs)

    if collect_features:
        all_features = np.vstack(all_features)
    else:
        all_features = None

    # 计算整体准确率
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

    # 保存分类报告
    report = classification_report(all_labels, all_preds, output_dict=True)

    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'routing_weights': all_routing_weights,
        'expert_outputs': all_expert_outputs,
        'features': all_features,
        'classification_report': report
    }


def plot_confusion_matrix(cm, class_names=None, save_path="confusion_matrix.png"):
    """绘制并保存混淆矩阵的热图"""
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
    plt.close()


def plot_classwise_accuracy(results, save_path="classwise_accuracy.png"):
    """绘制每个类别的分类准确率柱状图"""
    cm = results['confusion_matrix']
    correct_predictions = np.diag(cm)
    class_totals = cm.sum(axis=1)
    class_accuracy = correct_predictions / class_totals

    # 按准确率排序
    sorted_indices = np.argsort(class_accuracy)
    sorted_accuracy = class_accuracy[sorted_indices]
    sorted_class_names = [f"Class {i}" for i in sorted_indices]

    plt.figure(figsize=(12, 6))
    plt.bar(sorted_class_names, sorted_accuracy * 100, color="blue")
    plt.xlabel("Class ID")
    plt.ylabel("Accuracy (%)")
    plt.xticks(rotation=90, fontsize=8)
    plt.title("Class-wise Accuracy (Sorted)")
    plt.tight_layout()

    plt.savefig(save_path)
    print(f"Class-wise accuracy plot saved to {save_path}")
    plt.close()


def analyze_most_confused_classes(results, train_counts, top_n=10):
    """找出分类准确率最低的类别，并分析其误分类情况"""
    cm = results['confusion_matrix']
    num_classes = cm.shape[0]

    # 计算每个类别的样本总数
    class_totals = cm.sum(axis=1)
    correct_predictions = np.diag(cm)
    class_accuracy = correct_predictions / class_totals

    # 找到准确率最低的类别
    sorted_indices = np.argsort(class_accuracy)[:top_n]

    confused_classes_info = []

    print("\n=== Top Misclassified Classes ===")
    for idx in sorted_indices:
        class_name = f"Class {idx}"
        print(f"{class_name}: Train count = {train_counts.get(idx, 'N/A')}\t"
              f"Test count = {class_totals[idx]}\t"
              f"Accuracy = {class_accuracy[idx] * 100:.2f}%")

        # 找到误分类最多的目标类别
        misclassified = cm[idx].copy()
        misclassified[idx] = 0  # 去掉正确分类的部分
        most_confused_idx = np.argsort(misclassified)[-5:]  # 取误分类最多的前5个类

        class_info = {
            'class_id': idx,
            'accuracy': class_accuracy[idx] * 100,
            'test_count': int(class_totals[idx]),
            'train_count': train_counts.get(idx, 0),
            'confused_with': []
        }

        print(f"\tCommon misclassified as:")
        for mis_idx in reversed(most_confused_idx):
            mis_class_name = f"Class {mis_idx}"
            mis_count = cm[idx, mis_idx]
            mis_percent = (mis_count / class_totals[idx] * 100)
            print(f"\t- {mis_class_name}: {mis_count} samples ({mis_percent:.2f}%)")

            class_info['confused_with'].append({
                'class_id': int(mis_idx),
                'count': int(mis_count),
                'percentage': float(mis_percent)
            })

        confused_classes_info.append(class_info)
        print("")

    return confused_classes_info


def analyze_expert_specialization(results, num_classes, save_dir):
    """分析专家的专业化程度"""
    routing_weights = results['routing_weights']
    labels = results['labels']

    # 创建一个矩阵，记录每个类别对每个专家的偏好
    num_experts = routing_weights.shape[1]
    class_expert_matrix = np.zeros((num_classes, num_experts))

    # 统计每个类别对每个专家的总路由权重
    for i in range(len(labels)):
        label = labels[i]
        class_expert_matrix[label] += routing_weights[i]

    # 归一化，使得每个类别的专家偏好总和为1
    class_expert_matrix = class_expert_matrix / class_expert_matrix.sum(axis=1, keepdims=True)

    # 绘制类别-专家关联热图
    plt.figure(figsize=(12, 10))
    sns.heatmap(class_expert_matrix, cmap="YlGnBu")
    plt.xlabel("Expert ID")
    plt.ylabel("Class ID")
    plt.title("Class-Expert Routing Preference")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "class_expert_preference.png"))
    print(f"Class-expert preference heatmap saved")
    plt.close()

    # 找出每个类别最偏好的专家
    preferred_experts = np.argmax(class_expert_matrix, axis=1)
    expert_class_counts = {i: 0 for i in range(num_experts)}
    for expert_id in preferred_experts:
        expert_class_counts[expert_id] += 1

    # 绘制每个专家被多少类别偏好的柱状图
    plt.figure(figsize=(10, 6))
    plt.bar(range(num_experts), [expert_class_counts[i] for i in range(num_experts)])
    plt.xlabel("Expert ID")
    plt.ylabel("Number of Classes Preferring This Expert")
    plt.title("Expert Specialization")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "expert_specialization.png"))
    print("Expert specialization chart saved")
    plt.close()

    # 保存类别-专家偏好矩阵
    np.savetxt(os.path.join(save_dir, "class_expert_matrix.csv"), class_expert_matrix, delimiter=',')

    return {
        'class_expert_matrix': class_expert_matrix,
        'preferred_experts': preferred_experts,
        'expert_class_counts': expert_class_counts
    }


def compute_class_features(results, num_classes):
    """计算每个类别的特征向量"""
    if results['features'] is None:
        print("Warning: No feature data collected during evaluation")
        return None

    features = results['features']
    labels = results['labels']

    # 按类别分组特征
    class_features = {i: [] for i in range(num_classes)}
    for i, label in enumerate(labels):
        class_features[label].append(features[i])

    # 将列表转换为NumPy数组
    for cls in class_features:
        if class_features[cls]:
            class_features[cls] = np.array(class_features[cls])

    return class_features


def compute_class_similarity(class_features, save_path="class_similarity.png"):
    """计算类别特征的余弦相似度，并绘制热图"""
    if class_features is None:
        print("Warning: Cannot compute class similarity without features")
        return

    # 计算每个类别的平均特征向量
    class_means = {}
    for cls, features in class_features.items():
        if len(features) > 0:
            class_means[cls] = features.mean(axis=0).flatten()

    # 获取所有有效的类
    valid_classes = sorted(class_means.keys())

    # 构建特征矩阵，每一行是一个类的平均特征
    feature_matrix = np.array([class_means[cls] for cls in valid_classes])

    # 计算相似度矩阵
    similarity_matrix = cosine_similarity(feature_matrix)

    # 绘制相似度热图
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, cmap="coolwarm", xticklabels=valid_classes, yticklabels=valid_classes)
    plt.xlabel("Class ID")
    plt.ylabel("Class ID")
    plt.title("Class Feature Similarity (Cosine)")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Class similarity heatmap saved to {save_path}")
    plt.close()

    return similarity_matrix


def save_class_accuracy_results(results, train_counts, output_file="class_accuracy.csv"):
    """保存每个类别的准确率结果到CSV文件"""
    cm = results['confusion_matrix']
    correct_predictions = np.diag(cm)
    class_totals = cm.sum(axis=1)
    class_accuracy = correct_predictions / class_totals

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Class ID', 'Train Count', 'Test Count', 'Correct Count', 'Accuracy (%)'])

        for class_id in range(len(class_accuracy)):
            writer.writerow([
                class_id,
                train_counts.get(class_id, 0),
                int(class_totals[class_id]),
                int(correct_predictions[class_id]),
                f"{class_accuracy[class_id] * 100:.2f}"
            ])

    print(f"Class accuracy results saved to {output_file}")


def get_train_counts(train_loader):
    """获取训练集中每个类别的样本数量"""
    class_counts = {}
    for _, labels in train_loader:
        for label in labels.numpy():
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1
    return class_counts


def analyze_expert_agreement(results, save_dir):
    """分析专家之间的预测一致性"""
    expert_outputs = results['expert_outputs']  # [samples, num_experts, num_classes]

    # 获取每个专家的预测类别
    expert_predictions = np.argmax(expert_outputs, axis=2)  # [samples, num_experts]

    # 计算专家之间的预测一致性
    num_experts = expert_predictions.shape[1]
    agreement_matrix = np.zeros((num_experts, num_experts))

    for i in range(num_experts):
        for j in range(num_experts):
            # 计算专家i和专家j预测相同的样本比例
            agreement = np.mean(expert_predictions[:, i] == expert_predictions[:, j])
            agreement_matrix[i, j] = agreement

    # 绘制专家预测一致性热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(agreement_matrix, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.xlabel("Expert ID")
    plt.ylabel("Expert ID")
    plt.title("Expert Prediction Agreement")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "expert_agreement.png"))
    print(f"Expert agreement heatmap saved")
    plt.close()

    return agreement_matrix


def analyze_routing_distribution(results, save_dir):
    """分析路由权重的分布"""
    routing_weights = results['routing_weights']
    num_experts = routing_weights.shape[1]

    # 计算每个专家的平均路由权重
    expert_avg_weights = routing_weights.mean(axis=0)

    # 绘制专家平均路由权重柱状图
    plt.figure(figsize=(10, 6))
    plt.bar(range(num_experts), expert_avg_weights)
    plt.xlabel("Expert ID")
    plt.ylabel("Average Routing Weight")
    plt.title("Expert Utilization")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "expert_utilization.png"))
    print("Expert utilization chart saved")
    plt.close()

    # 计算路由的"熵"，表示路由的不确定性
    # 当一个样本的路由权重均匀分布时，熵最大；当全部集中在一个专家时，熵最小
    eps = 1e-10  # 避免log(0)
    routing_entropy = -np.sum(routing_weights * np.log(routing_weights + eps), axis=1)
    max_entropy = -np.log(1 / num_experts) * num_experts  # 最大可能熵
    normalized_entropy = routing_entropy / max_entropy

    # 绘制熵分布直方图
    plt.figure(figsize=(10, 6))
    plt.hist(normalized_entropy, bins=50)
    plt.xlabel("Normalized Routing Entropy")
    plt.ylabel("Count")
    plt.title("Routing Decision Uncertainty")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "routing_entropy.png"))
    print("Routing entropy histogram saved")
    plt.close()

    return {
        'expert_avg_weights': expert_avg_weights,
        'routing_entropy': routing_entropy
    }


def get_set_info(data_loader):
    """获取数据集中每个类别的样本数量"""
    class_counts = {}
    for _, labels in data_loader:
        unique_labels, counts = np.unique(labels.numpy(), return_counts=True)
        for label, count in zip(unique_labels, counts):
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += count
    return class_counts


def main():
    # 配置参数
    DATASET_PATH = "./data/AppClassNet/top200"
    RESULTS_PATH = "./results/AppClassNet/top200/MoE/2"
    OUTPUT_DIR = f"{RESULTS_PATH}/analysis"
    NUM_CLASSES = 200
    NUM_EXPERTS = 3
    BATCH_SIZE = 2048
    NUM_WORKERS = 12

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # # 查找最佳模型路径
    # best_model_files = [f for f in os.listdir(f"{RESULTS_PATH}/param") if f.startswith("best_model_")]
    # if not best_model_files:
    #     raise FileNotFoundError(f"No best_model found in {RESULTS_PATH}/param")
    #
    # # 选择准确率最高的模型
    # best_model = sorted(best_model_files, key=lambda x: float(x.split('_')[-1].split('.pth')[0]), reverse=True)[0]
    # checkpoint_path = f"{RESULTS_PATH}/param/{best_model}"
    # print(f"Selected checkpoint: {checkpoint_path}")

    # 加载数据
    train_x, train_y = load_data("train", DATASET_PATH)
    test_x, test_y = load_data("test", DATASET_PATH)

    # 创建数据加载器
    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # 获取训练集类别计数
    print("Calculating train set class counts...")
    train_counts = get_set_info(train_loader)

    # 加载模型
    model = load_model_from_checkpoint(None, NUM_CLASSES, NUM_EXPERTS, device)

    # 评估模型
    print("Evaluating model on test set...")
    start_time = time.time()
    results = evaluate_model(model, test_loader, device, OUTPUT_DIR, collect_features=True)
    evaluation_time = time.time() - start_time
    print(f"Evaluation completed in {evaluation_time:.2f} seconds")

    # 打印整体准确率
    print(f"Test Accuracy: {results['accuracy'] * 100:.2f}%")

    # 绘制混淆矩阵
    plot_confusion_matrix(
        results['confusion_matrix'],
        save_path=os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    )

    # 绘制每个类别的准确率
    plot_classwise_accuracy(
        results,
        save_path=os.path.join(OUTPUT_DIR, "classwise_accuracy.png")
    )

    # 分析误分类最多的类别
    confused_classes = analyze_most_confused_classes(results, train_counts, top_n=10)

    # 保存类别准确率结果
    save_class_accuracy_results(
        results,
        train_counts,
        output_file=os.path.join(OUTPUT_DIR, "class_accuracy.csv")
    )

    # 分析专家专业化程度
    expert_specialization = analyze_expert_specialization(results, NUM_CLASSES, OUTPUT_DIR)

    # 分析专家之间的预测一致性
    expert_agreement = analyze_expert_agreement(results, OUTPUT_DIR)

    # 分析路由权重分布
    routing_analysis = analyze_routing_distribution(results, OUTPUT_DIR)

    # 计算类别特征相似度
    print("Computing class features...")
    class_features = compute_class_features(results, NUM_CLASSES)

    if class_features is not None:
        similarity_matrix = compute_class_similarity(
            class_features,
            save_path=os.path.join(OUTPUT_DIR, "class_similarity.png")
        )

    print(f"Analysis completed. Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
