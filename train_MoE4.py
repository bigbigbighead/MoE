import os
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import glob
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from models.MoE import MoE4Model
from models.ResNet import resnet18

from utils.data_loading_mine import log_message, load_data, get_dataloaders, ROUTING_TYPE
from utils.load_model import load_pretrained_weights, load_checkpoint
from validate_model import validate_expert, validate_full_model, validate_router_accuracy, get_expert_indices

# 数据集路径
DATASET_PATH = "./data/AppClassNet/top200"
RESULTS_PATH = "./results/AppClassNet/top200/MoE/18"
# 预训练模型路径
PRETRAINED_RESNET18_PATH = "./results/AppClassNet/top200/ResNet/1/param/model_epoch_800.pth"  # 预训练ResNet18模型路径

# 确保结果目录存在
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(f"{RESULTS_PATH}/param", exist_ok=True)
os.makedirs(f"{RESULTS_PATH}/logs", exist_ok=True)

# 创建日志文件
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"{RESULTS_PATH}/logs/training_log_{current_time}.txt"

# 优化超参数
BATCH_SIZE = 2048  # 批次大小
EPOCHS_STAGE1 = 10  # 第一阶段训练轮数
LEARNING_RATE_STAGE1 = 0.001  # 第一阶段学习率
NUM_CLASSES = 200  # AppClassNet 类别数
CLASS_RANGES = [(0, 5), (6, 26), (27, 199)]
NUM_EXPERTS = 3  # MoE专家头数量
NUM_WORKERS = 2  # 数据加载的worker数量
PIN_MEMORY = True  # 确保启用pin_memory
PREFETCH_FACTOR = 8  # 增加预取因子
# 自动混合精度训练配置
USE_AMP = True  # 启动自动混合精度训练


# 第一阶段训练：单独训练每个专家
def train_stage1(model, train_loaders, val_loaders, test_loaders, device, resume_training=False):
    """
    第一阶段训练：分别训练每个专家
    修改后的逻辑：外层循环是专家索引，内层循环是每个专家的完整训练周期
    """
    log_message(f"开始专家训练...")
    writer = SummaryWriter(os.path.join(RESULTS_PATH, 'logs', 'stage1'))

    # 冻结backbone参数
    for param in model.backbone.parameters():
        param.requires_grad = False

    log_message("已冻结backbone参数，仅训练专家分类器")

    # 创建混合精度训练的scaler
    scaler = GradScaler() if USE_AMP else None

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 外层循环：每个专家
    for expert_idx in range(len(model.experts)):
        log_message(f"===== 开始训练专家 {expert_idx + 1}/{len(model.experts)} =====")
        start_class, end_class = model.class_ranges[expert_idx]

        # 只优化当前专家的参数
        optimizer = optim.Adam(model.experts[expert_idx].parameters(), lr=LEARNING_RATE_STAGE1)

        # 添加恢复训练功能
        start_epoch = 0
        best_val_acc = 0.0

        if resume_training:
            expert_checkpoint_path = os.path.join(RESULTS_PATH, 'param', f'stage1_expert{expert_idx}_latest.pth')
            if os.path.exists(expert_checkpoint_path):
                checkpoint = torch.load(expert_checkpoint_path)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    # 只加载当前专家的参数
                    model_dict = model.state_dict()
                    expert_dict = {k: v for k, v in checkpoint['model_state_dict'].items()
                                   if k.startswith(f'experts.{expert_idx}') or k.startswith(
                            f'module.experts.{expert_idx}')}
                    model_dict.update(expert_dict)
                    model.load_state_dict(model_dict)

                    if 'optimizer_state_dict' in checkpoint:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                    if 'epoch' in checkpoint:
                        start_epoch = checkpoint['epoch'] + 1

                    if 'best_val_acc' in checkpoint:
                        best_val_acc = checkpoint['best_val_acc']

                    log_message(
                        f"已加载专家{expert_idx}的检查点，从第{start_epoch}轮继续训练，当前最佳验证准确率: {best_val_acc:.2f}%")

        # 获取当前专家的数据加载器
        train_loader = train_loaders[expert_idx]
        val_loader_expert = val_loaders[expert_idx]
        test_loader_expert = test_loaders[expert_idx]
        # 内层循环：该专家的完整训练周期
        for epoch in range(start_epoch, EPOCHS_STAGE1):
            epoch_start_time = time.time()
            log_message(f"专家{expert_idx} - Epoch {epoch + 1}/{EPOCHS_STAGE1}")
            model.train()

            # 训练一个epoch
            epoch_loss = 0
            correct = 0
            total = 0
            batch_train_time = 0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                batch_start_time = time.time()
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)  # 更快地重置梯度

                # 使用混合精度训练
                if USE_AMP and scaler is not None:
                    with autocast():
                        # 前向传播
                        features = model.backbone(inputs)
                        outputs = model.experts[expert_idx](features)

                        # 计算损失
                        loss = criterion(outputs, targets)

                    # 使用scaler进行反向传播和优化
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # 前向传播
                    features = model.backbone(inputs)
                    outputs = model.experts[expert_idx](features)

                    # 计算损失
                    loss = criterion(outputs, targets)

                    # 反向传播与优化
                    loss.backward()
                    optimizer.step()

                batch_time_taken = time.time() - batch_start_time
                batch_train_time += batch_time_taken

                # 统计
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            # 记录每个专家的训练损失和准确率
            avg_loss = epoch_loss / len(train_loader)
            accuracy = correct / total
            epoch_time_taken = time.time() - epoch_start_time
            writer.add_scalar(f'expert{expert_idx}/train_loss', avg_loss, epoch)
            writer.add_scalar(f'expert{expert_idx}/train_accuracy', accuracy, epoch)
            log_message(
                f"  专家{expert_idx} - 训练损失: {avg_loss:.4f}, 准确率: {accuracy:.4f}, 耗时: {epoch_time_taken:.2f}s (Avg Batch: {batch_train_time / len(train_loader):.2f}s)")

            # 使用专家对应的验证集验证当前专家的性能
            valid_start_time = time.time()
            val_loss, val_accuracy = validate_expert(model, val_loader_expert, criterion, expert_idx, device)
            valid_end_time = time.time()
            writer.add_scalar(f'expert{expert_idx}/val_loss', val_loss, epoch)
            writer.add_scalar(f'expert{expert_idx}/val_accuracy', val_accuracy, epoch)
            log_message(
                f"  专家{expert_idx} - 验证损失: {val_loss:.4f}, 准确率: {val_accuracy:.4f}, 耗时: {valid_end_time - valid_start_time:.2f}s")

            # 使用专家对应的测试集测试当前专家的性能
            test_start_time = time.time()
            test_loss, test_accuracy = validate_expert(model, test_loader_expert, criterion, expert_idx, device)
            test_end_time = time.time()
            writer.add_scalar(f'expert{expert_idx}/test_loss', test_loss, epoch)
            writer.add_scalar(f'expert{expert_idx}/test_accuracy', test_accuracy, epoch)
            log_message(
                f"  专家{expert_idx} - 测试损失: {test_loss:.4f}, 准确率: {test_accuracy:.4f}, 耗时: {test_end_time - test_start_time:.2f}s")

            # 保存当前专家的最新检查点
            checkpoint_path = os.path.join(RESULTS_PATH, 'param', f'stage1_expert{expert_idx}_latest.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'best_val_acc': max(best_val_acc, val_accuracy)
            }, checkpoint_path)

            # 如果是最佳性能，保存最佳检查点
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                best_checkpoint_path = os.path.join(RESULTS_PATH, 'param', f'stage1_expert{expert_idx}_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                    'best_val_acc': best_val_acc
                }, best_checkpoint_path)
                log_message(f"  已保存专家{expert_idx}的最佳模型，验证准确率: {best_val_acc:.4f}")

            # 定期保存阶段性检查点
            if (epoch + 1) % 10 == 0 or (epoch + 1) == EPOCHS_STAGE1:
                epoch_checkpoint_path = os.path.join(RESULTS_PATH, 'param',
                                                     f'stage1_expert{expert_idx}_epoch{epoch + 1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                    'best_val_acc': best_val_acc
                }, epoch_checkpoint_path)
                log_message(f"  已保存专家{expert_idx}的Epoch {epoch + 1}检查点")

        # 关闭该专家的数据加载器
        del train_loader, val_loader_expert, test_loader_expert
        # 该专家训练完成
        log_message(f"专家{expert_idx}训练完成，最佳验证准确率: {best_val_acc:.4f}")

    # 所有专家训练完成后，保存完整模型
    final_checkpoint_path = os.path.join(RESULTS_PATH, 'param', 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
    }, final_checkpoint_path)
    log_message(f"所有专家训练完成，完整模型已保存到 {final_checkpoint_path}")
    writer.close()

    return model


# 修改验证函数，确保与新的推理逻辑一致
def validate_expert(model, val_loader, criterion, expert_idx, device):
    """验证单个专家的性能"""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    start_class, end_class = model.module.class_ranges[expert_idx] if hasattr(model, 'module') else model.class_ranges[
        expert_idx]

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # 获取特征
            features = model.module.backbone(inputs) if hasattr(model, 'module') else model.backbone(inputs)

            # 获取专家输出
            outputs = model.module.experts[expert_idx](features) if hasattr(model, 'module') else model.experts[
                expert_idx](features)

            # 计算损失
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return val_loss / len(val_loader), correct / total


# 修改整体模型的评估函数，确保使用新的推理逻辑
def evaluate_ensemble_model(model, val_loader, test_loader, device):
    """评估整体模型性能，使用新的推理逻辑（拼接专家输出）"""
    log_message("开始评估整体模型性能...")
    model.eval()

    criterion = nn.CrossEntropyLoss()

    # 评估验证集性能
    val_start_time = time.time()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # 获取模型预测（使用inference方法确保用拼接而非相加）
            logits = model.inference(inputs)
            loss = criterion(logits, targets)

            val_loss += loss.item()
            _, predicted = logits.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()

    val_accuracy = val_correct / val_total
    val_loss = val_loss / len(val_loader)
    val_time = time.time() - val_start_time

    log_message(f"验证集性能 - 损失: {val_loss:.4f}, 准确率: {val_accuracy:.4f}, 耗时: {val_time:.2f}s")

    # 评估测试集性能
    test_start_time = time.time()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # 获取模型预测
            logits = model.inference(inputs)
            loss = criterion(logits, targets)

            test_loss += loss.item()
            _, predicted = logits.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()

    test_accuracy = test_correct / test_total
    test_loss = test_loss / len(test_loader)
    test_time = time.time() - test_start_time

    log_message(f"测试集性能 - 损失: {test_loss:.4f}, 准确率: {test_accuracy:.4f}, 耗时: {test_time:.2f}s")

    return val_accuracy, test_accuracy


if __name__ == "__main__":
    # 设置NUMA绑定和多线程优化
    if torch.cuda.is_available():
        # 启用CUDA性能优化
        torch.backends.cudnn.benchmark = True

    # 如果使用CPU，则设置OpenMP线程数
    if not torch.cuda.is_available():
        torch.set_num_threads(NUM_WORKERS)  # 设置适当的线程数

    # 记录训练开始信息和配置信息
    log_message(f"=== 训练开始于 {current_time} ===")
    log_message(f"BatchSize: {BATCH_SIZE}")
    log_message(f"学习率: {LEARNING_RATE_STAGE1}, 训练轮数: {EPOCHS_STAGE1}")
    log_message(f"数据集路径: {DATASET_PATH}")
    log_message(f"结果保存路径: {RESULTS_PATH}")
    log_message(f"使用设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    log_message(f"自动混合精度: {USE_AMP}")
    log_message(f"工作进程数: {NUM_WORKERS}")
    log_message(f"专家数量: {NUM_EXPERTS}")
    log_message(f"路由类型: {ROUTING_TYPE}")
    log_message(f"类别范围: {CLASS_RANGES}")
    log_message(f"预训练模型路径: {PRETRAINED_RESNET18_PATH}")

    # 配置数据并行训练，利用多个GPU
    multi_gpu = torch.cuda.device_count() > 1
    log_message(f"使用GPU数量: {torch.cuda.device_count()}")

    # 添加继续训练选项
    RESUME_TRAINING = True  # 设置是否从检查点继续训练

    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    train_loaders, val_loaders, test_loaders, full_train_loader, full_val_loader, full_test_loader = get_dataloaders(
        CLASS_RANGES)

    # 从训练数据中获取输入形状
    sample_input, _ = next(iter(full_train_loader))
    input_channels = sample_input.shape[1]
    input_height = sample_input.shape[2]
    input_width = sample_input.shape[3]

    log_message(f"模型输入形状: 通道={input_channels}, 高度={input_height}, 宽度={input_width}")

    # 创建模型
    model = MoE4Model(
        total_classes=NUM_CLASSES,
        class_ranges=CLASS_RANGES,
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width
    )

    # 加载预训练ResNet18权重
    model = load_pretrained_weights(model, PRETRAINED_RESNET18_PATH)

    # 如果有多GPU，使用DataParallel
    if multi_gpu:
        log_message("启用多GPU并行训练")
        model = nn.DataParallel(model)

    model.to(device)

    # 训练专家
    model = train_stage1(model, train_loaders, val_loaders, test_loaders, device, resume_training=RESUME_TRAINING)

    # 评估整体模型
    val_accuracy, test_accuracy = evaluate_ensemble_model(model, full_val_loader, full_test_loader, device)

    # 记录最终结果
    log_message(f"训练完成！")
    log_message(f"最终验证集准确率: {val_accuracy:.4f}")
    log_message(f"最终测试集准确率: {test_accuracy:.4f}")
    log_message(f"=== 训练结束于 {datetime.datetime.now().strftime('%Y.%m.%d_%H:%M:%S')} ===")
