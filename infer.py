from datetime import time
import yaml
import os
import numpy as np
import torch
import time
import csv
import matplotlib.pyplot as plt
import utils.train_and_test
from utils.data_loading import import_data, get_loaders
from utils.model_loading import load_model
from sklearn.metrics import confusion_matrix


def load_model_parameters():
    """
    构建并加载训练好的模型,读取测试数据集
    """
    print(torch.cuda.is_available())
    num_gpus = torch.cuda.device_count()
    print(num_gpus)
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    config_dir = "configuration/config_1.yml"
    with open(config_dir, "r") as config_file:
        configuration = yaml.safe_load(config_file)

    os.environ["CUDA_VISIBLE_DEVICES"] = configuration["gpu"]

    # set experiment folder
    xp_dir = (
            "./results/"
            + str(configuration["dataset"])
            + "/"
            + str(configuration["model_name"])
            + "/"
            + str(configuration["experiment_run"])
            + "/"
    )
    # Load dataset
    X_train, y_train, X_validation, y_validation, X_test, y_test = import_data(
        configuration["dataset"], xp_dir, val_split=configuration["validation_split"]
    )

    # Initial values
    change = True
    epoch_update = 0
    nbclass = len(np.unique(y_train))
    num_prototypes = nbclass * configuration["prototypes"]
    prototype_shape = (
        num_prototypes,
        configuration["base_architecture_last_filters"],
        configuration["prototype_size"][0],
        configuration["prototype_size"][1],
    )
    prototype_class_identity = torch.zeros(num_prototypes, nbclass)
    for j in range(num_prototypes):
        prototype_class_identity[j, j // configuration["prototypes"]] = 1
    num_prototypes_per_class = {}
    for i in range(nbclass):
        num_prototypes_per_class[i] = [i]

    # Build and load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(
        configuration,
        prototype_shape,
        prototype_class_identity,
        num_prototypes_per_class,
        size=[X_train.shape[2], X_train.shape[3]],
        nbclass=nbclass,
    )
    # torch.save(
    #     obj=model,
    #     f=os.path.join(
    #         xp_dir,
    #         (configuration["model_name"] + "_{0:.2f}.pth").format(accu_test * 100),
    #     ),
    # )
    checkpoint_name = "88.21"
    checkpoint_path = os.path.join(xp_dir, "param/", (configuration["model_name"] + "_" + checkpoint_name + ".pth"))
    # checkpoint = torch.load(checkpoint_path, device)
    model = torch.load(checkpoint_path)
    model = model.to(device)
    # model_multi = torch.nn.DataParallel(model)

    # load data
    loader_train, loader_validation, loader_test = get_loaders(
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        batch_size=configuration["batch_size"],
        shuffle=True,
    )
    # test_result = xp_dir + "class_accuracy.csv"
    # detailed_inference_test(model, loader_test, xp_dir)
    return model, loader_train, loader_test, xp_dir


def test_model(model, loader_test, xp_dir):
    """
    测试模型的推理时间和准确度
    """
    model.eval()
    start_time = time.time()
    n_examples = 0
    n_correct = 0

    with torch.no_grad():
        for sample, label in loader_test:
            input = sample.cuda()
            target = label.cuda()
            target = target.to(dtype=torch.long)
            output, _ = model(input)
            _, predicted = torch.max(output.data, 1)

            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

    end_time = time.time()
    total_time = end_time - start_time
    test_sample_num = len(loader_test.dataset)  # 获取的是测试集的总样本数
    # test_sample_num2 = len(loader_test)#获取的是批次数(batches)
    inference_speed = total_time / test_sample_num
    inference_throughput = test_sample_num / total_time
    text = (
            "\tTime: \t{0}\n".format(total_time)
            + "\tTest set size: \t{0}\n".format(test_sample_num)
            + "\tAccuracy: \t\t{0}%\n".format(n_correct / n_examples * 100)
            + "\tInference speed: \t{0} μs/sample\n".format(inference_speed * 1e6)
            + "\tInference throughput: \t{0} samples/s\n".format(inference_throughput)
    )
    print(text)
    with open(xp_dir + "inference_test.log", "a") as f:
        f.write(text)


def save_data(class_sample_count, class_correct_count, xp_dir, class_accuracy_data="class_accuracy.csv"):
    """
    保存每个类别的样本数、准确率到 CSV 文件

    Parameters
    ----------
    class_sample_count: dict
        每个类别的样本数量
    class_correct_count: dict
        每个类别的正确分类数量
    file_path: str, optional
        保存文件的路径
    """
    # 计算每个类别的准确率
    class_accuracy = {class_id: (class_correct_count[class_id] / class_sample_count[class_id]) * 100
                      for class_id in class_sample_count}
    file_path = os.path.join(xp_dir, class_accuracy_data)
    # 将数据写入 CSV 文件
    with open(file_path, mode="w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "accuracy", "sample_count"])  # 写入列标题
        for class_id in class_sample_count:
            accuracy = class_accuracy[class_id]
            sample_count = class_sample_count[class_id]
            writer.writerow([class_id, accuracy, sample_count])  # 写入每一行数据

    print(f"Class accuracy data saved to {file_path}")


def read_data(xp_dir, class_accuracy_data="class_accuracy.csv"):
    """
    从 CSV 文件中读取类别的准确率数据

    Parameters
    ----------
    file_path: str
        读取文件的路径

    Returns
    -------
    class_accuracy_data: list of tuples
        包含 class_id、accuracy 和 sample_count 的数据
    """
    file_path = os.path.join(xp_dir, class_accuracy_data)
    class_accuracy = []
    with open(file_path, mode="r") as f:
        reader = csv.reader(f)
        next(reader)  # 跳过标题行
        for row in reader:
            class_id, accuracy, sample_count = int(row[0]), float(row[1]), int(row[2])
            class_accuracy.append((class_id, accuracy, sample_count))

    return class_accuracy


def save_infer_info(train_class_count, test_class_count, test_correct_count, xp,
                    result_file="test_inference_result.csv"):
    file_path = os.path.join(xp, result_file)

    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Class', 'Train set size', 'Test set size', 'Test correct count', 'Test accuracy'])
        for i in range(len(train_class_count)):
            accuracy = test_correct_count[i] / test_class_count[i]
            writer.writerow([i, train_class_count[i], test_class_count[i], test_correct_count[i], accuracy])
    print(f"详细结果已保存至: {file_path}")


def get_set_info(loader_train):
    """
    获取数据集的每一个类的数量
    """
    n_examples = 0
    class_sample_count = {}
    for _, labels in loader_train:
        # input = sample.cuda()
        target = labels.cuda().to(dtype=torch.long)
        n_examples += target.size(0)
        for i in range(target.size(0)):
            label = target[i].item()
            if label not in class_sample_count:
                class_sample_count[label] = 0
            class_sample_count[label] += 1

    return class_sample_count


def plot_class_accuracy(xp_dir, class_accuracy):
    """
    绘制类别的准确率柱状图

    Parameters
    ----------
    class_accuracy_data: list of tuples
        包含 class_id、accuracy 和 sample_count 的数据
    """
    # 按照 class_id 排序
    sorted_data = sorted(class_accuracy, key=lambda x: x[0])
    class_ids = [x[0] for x in sorted_data]
    accuracies = [x[1] for x in sorted_data]

    # 绘制柱状图
    plt.figure(figsize=(10, 6))
    plt.bar(class_ids, accuracies, color='skyblue')
    plt.xlabel('Class ID')
    plt.ylabel('Accuracy (%)')
    plt.title('Class Accuracy per Category')
    plt.xticks(class_ids)
    plt.tight_layout()
    plt.savefig(xp_dir + "class_accuracy.png")
    print(f"Plot saved as {xp_dir}class_accuracy.png")
    plt.show()


def show_data(xp_dir, class_accuracy_data="class_accuracy.csv"):
    """
        读取保存的测试集每个类别的准确度结果并画图展示出来
    """
    # 读取数据并绘制柱状图
    class_accuracy_data = read_data(xp_dir, class_accuracy_data)
    plot_class_accuracy(xp_dir, class_accuracy_data)


def detailed_inference_test(model, lodaer_train, loader_test, xp_dir=None, class_accuracy_data="class_accuracy.csv"):
    """
        进行详细的测试集推理，记录每个类别的样本数、准确率、以及误分类样本信息
        保存每个类别样本数、正确分类样本数到class_accuracy_data

        Parameters
        ----------
        model: torch.nn.Module
            训练好的模型
        loader_test: DataLoader
            测试集数据加载器
        xp_dir: string, optional
            结果存储路径
        log: function, optional
            日志输出函数

        Returns
        -------
        accuracy: float
            测试集准确率
        """
    print("Analysing test set:")
    model.eval()  # 设置评估模式，避免 BatchNorm 和 Dropout 影响结果

    start_time = time.time()
    n_examples = 0
    n_correct = 0

    # 用于记录每个类别的样本数和预测正确的样本数
    class_sample_count = {}  # 测试集每个类的数量
    class_correct_count = {}
    misclassified_samples = []

    # 真实标签和预测结果，用于生成 confusion matrix
    all_labels = []
    all_predictions = []

    with torch.no_grad():  # 禁用梯度计算，加速推理
        for sample, label in loader_test:
            input = sample.cuda()
            target = label.cuda().to(dtype=torch.long)

            output, _ = model(input)  # 仅进行前向传播
            _, predicted = torch.max(output.data, 1)  # 获取预测类别

            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            # 统计每个类别的样本数和正确分类的样本数
            for i in range(target.size(0)):
                label = target[i].item()
                pred = predicted[i].item()

                if label not in class_sample_count:
                    class_sample_count[label] = 0
                    class_correct_count[label] = 0
                class_sample_count[label] += 1
                if label == pred:
                    class_correct_count[label] += 1
                else:
                    misclassified_samples.append((i, label, pred))  # 记录误分类样本
            # 收集所有标签和预测结果，用于计算混淆矩阵
            all_labels.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # 计算推理时间
    total_time = time.time() - start_time
    test_sample_num = len(loader_test.dataset)
    inference_speed = total_time / test_sample_num  # 秒/样本
    inference_throughput = test_sample_num / total_time  # 样本/秒

    # 计算每个类别的准确率
    class_accuracy = {}
    for class_id in class_sample_count:
        class_accuracy[class_id] = class_correct_count[class_id] / class_sample_count[class_id] * 100

    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    np.savetxt(xp_dir + "confusion_matrix.csv", cm, delimiter=',', fmt="%d")
    # 输出详细的测试信息
    text = (
        f"\tTime: \t{total_time:.4f} sec\n"
        f"\tTest set size: \t{test_sample_num}\n"
        f"\tAccuracy: \t\t{(n_correct / n_examples) * 100:.2f}%\n"
        f"\tInference speed: \t{inference_speed * 1e6:.6f} μs/sample\n"
        f"\tInference throughput: \t{inference_throughput:.2f} samples/s\n"
    )

    print(text)

    # 排序类别并输出每个类别的准确率和样本数量
    sorted_class_ids = sorted(class_sample_count.keys())  # 按类别ID从小到大排序
    # 按照类别准确率从小到大排序
    sorted_class_accuracy = sorted(class_accuracy.items(), key=lambda x: x[1])  # 排序，按准确率从小到大

    for class_id in sorted_class_ids:
        class_accuracy_str = f"Class {class_id} Accuracy: {class_accuracy[class_id]:.2f}%"
        class_sample_count_str = f"Test Sample count: {class_sample_count[class_id]}"
        print(f"{class_accuracy_str}, {class_sample_count_str}")
    print("\n**********Class sorted by accuracy ***************\n")
    for class_id, accuracy in sorted_class_accuracy:
        class_sample_count_str = f"Test Sample count: {class_sample_count[class_id]}"
        print(f"Class {class_id} Accuracy: {accuracy:.2f}%, {class_sample_count_str}")

    # # 输出混淆矩阵
    # print("Confusion Matrix:")
    # print(cm)
    #
    # # 记录误分类的样本（只输出前 10 个误分类样本为例）
    # print("Misclassified Samples (index, true label, predicted label):")
    # for sample in misclassified_samples[:10]:  # 只输出前 10 个误分类样本
    #     print(f"Sample {sample[0]}, True: {sample[1]}, Pred: {sample[2]}")

    # 记录到日志文件
    if xp_dir:
        with open(xp_dir + "detailed_inference_test.log", "a") as f:
            f.write(text + "\n")
            for class_id in sorted_class_ids:
                f.write(
                    f"Class {class_id} Accuracy: {class_accuracy[class_id]:.2f}%, Test Sample count: {class_sample_count[class_id]}\n")
            f.write("\n**********Class sorted by accuracy ***************\n")
            for class_id, accuracy in sorted_class_accuracy:
                f.write(
                    f"Class {class_id} Accuracy: {accuracy:.2f}%, Test Sample count: {class_sample_count[class_id]}\n")
            f.write("Confusion Matrix:\n")
            f.write(np.array_str(cm) + "\n")
            f.write("Misclassified Samples:\n")
            for sample in misclassified_samples[:10]:
                f.write(f"Sample {sample[0]}, True: {sample[1]}, Pred: {sample[2]}\n")
    # 保存数据到文件
    save_data(class_sample_count, class_correct_count, xp_dir, class_accuracy_data)

    train_class_count = get_set_info(loader_train)
    save_infer_info(train_class_count, class_sample_count, class_correct_count, xp_dir)
    # 读取数据并绘制柱状图
    accuracy_data = read_data(xp_dir, class_accuracy_data)
    plot_class_accuracy(xp_dir, accuracy_data)
    return n_correct / n_examples, total_time, cm


if __name__ == "__main__":
    class_accuracy_data = "class_accuracy.csv"
    # 加载模型和数据集
    model, loader_train, loader_test, xp_dir = load_model_parameters()

    # # 测试模型测试集准确率和速度
    # test_model(model, loader_test, xp_dir)

    # 分析模型测试集上的详细数据
    detailed_inference_test(model, loader_train, loader_test, xp_dir, class_accuracy_data)

    # 画图展示测试集上的效果
    # show_data(xp_dir, class_accuracy_data)
