import numpy as np
import os
import argparse
import yaml
from utils.data_loading import import_data, get_loaders
from sklearn.model_selection import train_test_split
from collections import Counter


def dataset_class_partition(selected_classes, xp_dir, file_name):
    """按原数据集的类别划分成新数据集"""
    x_file_path = os.path.join(xp_dir, file_name + '_x.npy')
    y_file_path = os.path.join(xp_dir, file_name + '_y.npy')
    dataset_x = np.load(x_file_path)
    dataset_y = np.load(y_file_path)
    selected_data_label = np.isin(dataset_y, selected_classes)
    new_x_data = dataset_x[selected_data_label]
    new_y_data = dataset_y[selected_data_label]
    np.save(os.path.join(xp_dir, 'new_' + file_name + '_x.npy'), new_x_data)
    np.save(os.path.join(xp_dir, 'new_' + file_name + '_y.npy'), new_y_data)


def class_balanced_partition(xp_dir, file_name, target_count=None):
    x_file_path = os.path.join(xp_dir, file_name + '_x.npy')
    y_file_path = os.path.join(xp_dir, file_name + '_y.npy')
    dataset_x = np.load(x_file_path)
    dataset_y = np.load(y_file_path)
    # 获取每个类别的样本索引
    unique_classes = np.unique(dataset_y)
    class_indices = {cls: np.where(dataset_y == cls)[0] for cls in unique_classes}

    # 如果没有指定目标样本数，则选择最小类别的样本数量作为目标
    if target_count is None:
        target_count = min(len(indices) for indices in class_indices.values())

    # 对每个类别进行下采样，确保每个类别的样本数相同
    sampled_indices = []
    for cls, indices in class_indices.items():
        # 随机选择目标数量的样本
        sampled_indices.append(np.random.choice(indices, target_count, replace=False))

    # 合并所有采样后的索引
    balanced_indices = np.concatenate(sampled_indices)

    # 生成平衡后的数据集
    balanced_x = dataset_x[balanced_indices]
    balanced_y = dataset_y[balanced_indices]
    np.save(os.path.join(xp_dir, 'class_balanced_' + file_name + '_x.npy'), balanced_x)
    np.save(os.path.join(xp_dir, 'class_balanced_' + file_name + '_y.npy'), balanced_y)
    # 重新统计新训练集的类别分布
    new_class_counts = Counter(balanced_y)
    print("新训练集类别样本数量:")
    for class_id, count in new_class_counts.items():
        print(f"类别 {class_id}: {count} 样本")
    return balanced_x, balanced_y


def dataset_count_partition(high_acc_classes, xp_dir, file_name, count=1400):
    """把数据集的特定类的数量降到count个，其余类不变，变成新的数据集"""
    x_file_path = os.path.join(xp_dir, file_name + '_x.npy')
    y_file_path = os.path.join(xp_dir, file_name + '_y.npy')
    dataset_x = np.load(x_file_path)
    dataset_y = np.load(y_file_path)
    new_dataset_x = []
    new_dataset_y = []
    print(f"训练集: train_x = {dataset_x.shape}, train_y = {dataset_y.shape}")
    for class_id in np.unique(dataset_y):
        # 获取该类别的索引
        class_indices = np.where(dataset_y == class_id)[0]  # 获取该类别所有样本的索引
        if class_id in high_acc_classes and len(class_indices) > count:
            selected_indices = np.random.choice(class_indices, count)
        else:
            selected_indices = class_indices
        new_dataset_x.append(dataset_x[selected_indices])
        new_dataset_y.append(dataset_y[selected_indices])
    new_dataset_x = np.concatenate(new_dataset_x, axis=0)
    new_dataset_y = np.concatenate(new_dataset_y, axis=0)
    np.save(os.path.join(xp_dir, 'new_' + file_name + '_x.npy'), new_dataset_x)
    np.save(os.path.join(xp_dir, 'new_' + file_name + '_y.npy'), new_dataset_y)

    print(f"new训练集: train_x = {new_dataset_x.shape}, train_y = {new_dataset_y.shape}")
    # 重新统计新训练集的类别分布
    new_class_counts = Counter(new_dataset_y)
    print("新训练集类别样本数量:")
    for class_id, count in new_class_counts.items():
        print(f"类别 {class_id}: {count} 样本")


def dataset_count_new_partition(high_acc_classes, xp_dir, file_name, count=1400):
    """把数据集的特定类的数量降到count个，并将其独立划出来变成新的数据集"""
    x_file_path = os.path.join(xp_dir, file_name + '_x.npy')
    y_file_path = os.path.join(xp_dir, file_name + '_y.npy')
    dataset_x = np.load(x_file_path)
    dataset_y = np.load(y_file_path)
    new_dataset_x = []
    new_dataset_y = []
    print(f"数据集: x = {dataset_x.shape}, y = {dataset_y.shape}")
    for class_id in np.unique(dataset_y):
        # 获取该类别的索引
        class_indices = np.where(dataset_y == class_id)[0]  # 获取该类别所有样本的索引
        if class_id in high_acc_classes and len(class_indices) > count:
            print(class_id)
            selected_indices = np.random.choice(class_indices, count)
            new_dataset_x.append(dataset_x[selected_indices])
            new_dataset_y.append(dataset_y[selected_indices])
    new_dataset_x = np.concatenate(new_dataset_x, axis=0)
    new_dataset_y = np.concatenate(new_dataset_y, axis=0)
    np.save(os.path.join(xp_dir, 'new_' + file_name + '_x.npy'), new_dataset_x)
    np.save(os.path.join(xp_dir, 'new_' + file_name + '_y.npy'), new_dataset_y)

    print(f"new数据集: x = {new_dataset_x.shape}, y = {new_dataset_y.shape}")
    # # 重新统计新训练集的类别分布
    # new_class_counts = Counter(new_dataset_y)
    # print("新训练集类别样本数量:")
    # for class_id, count in new_class_counts.items():
    #     print(f"类别 {class_id}: {count} 样本")


def print_class_distribution(xp_dir, file_name):
    x_file_path = os.path.join(xp_dir, file_name + '_x.npy')
    y_file_path = os.path.join(xp_dir, file_name + '_y.npy')
    new_x_file_path = os.path.join(xp_dir, 'new_' + file_name + '_x.npy')
    new_y_file_path = os.path.join(xp_dir, 'new_' + file_name + '_y.npy')
    x_data = np.load(x_file_path)
    y_data = np.load(y_file_path)
    new_x_data = np.load(new_x_file_path)
    new_y_data = np.load(new_y_file_path)
    print("x_shape:", x_data.shape)
    print("y_shape:", y_data.shape)
    print("new_x_shape:", new_x_data.shape)
    print("new_y_shape:", new_y_data.shape)


if __name__ == '__main__':
    low_acc_classes_test = [193, 179, 184, 176, 199, 169, 191, 162, 189]
    low_acc_classes = [193, 179, 140, 184, 59, 176, 82, 199, 94, 169, 191, 162, 189, 134, 133]
    xp_dir = "./data/AppClassNet/top200/"
    high_acc_class_test = [0, 21, 7]
    high_acc_class = [0, 21, 7, 60, 26, 136, 48, 5, 10, 11, 91, 23, 112, 117, 104, 34, 40, 35, 56, 24, 77, 124, 25,
                      45]
    print(len(high_acc_class))
    # dataset_class_partition(low_acc_classes_test, xp_dir, "train")
    # dataset_class_partition(low_acc_classes_test, xp_dir, "valid")
    # dataset_class_partition(low_acc_classes_test, xp_dir, "test")

    count = 1400
    # data_count_partition(high_acc_class_test, xp_dir, "train", count)
    # print_class_distribution(xp_dir, "train")
    # dataset_count_new_partition(high_acc_class, xp_dir, "train", count)
    # dataset_class_partition(high_acc_class, xp_dir, "valid")
    # dataset_class_partition(high_acc_class, xp_dir, "test")

    class_balanced_partition(xp_dir, "train")
    # Class 193 Accuracy: 0.00%, Sample count: 1631
    # Class 179 Accuracy: 0.05%, Sample count: 2000
    # Class 140 Accuracy: 1.53%, Sample count: 3270
    # Class 184 Accuracy: 3.94%, Sample count: 1803
    # Class 59 Accuracy: 6.19%, Sample count: 15883
    # Class 176 Accuracy: 6.98%, Sample count: 2035
    # Class 82 Accuracy: 9.43%, Sample count: 9913
    # Class 199 Accuracy: 9.95%, Sample count: 1547
    # Class 94 Accuracy: 11.16%, Sample count: 7501
    # Class 169 Accuracy: 13.51%, Sample count: 2295
    # Class 191 Accuracy: 13.68%, Sample count: 1659
    # Class 162 Accuracy: 13.96%, Sample count: 2593
    # Class 189 Accuracy: 14.26%, Sample count: 1704
    # Class 134 Accuracy: 14.46%, Sample count: 3533
    # Class 133 Accuracy: 14.71%, Sample count: 3636
    # Class 180 Accuracy: 15.06%, Sample count: 1945
    # Class 130 Accuracy: 16.23%, Sample count: 3906
    # Class 118 Accuracy: 17.71%, Sample count: 5189
    # Class 109 Accuracy: 18.01%, Sample count: 6495
    # Class 57 Accuracy: 18.02%, Sample count: 16266
    # Class 155 Accuracy: 18.14%, Sample count: 2751
    # Class 119 Accuracy: 18.33%, Sample count: 5029
    # Class 137 Accuracy: 18.59%, Sample count: 3469
    # Class 183 Accuracy: 18.76%, Sample count: 1807
