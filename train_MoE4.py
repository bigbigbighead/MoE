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

from utils.data_loading_mine import log_message, load_data, get_dataloaders

# 数据集路径
DATASET_PATH = "./data/AppClassNet/top200"
RESULTS_PATH = "./results/AppClassNet/top200/MoE/4"
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
EPOCHS_STAGE1 = 300  # 第一阶段训练轮数
EPOCHS_STAGE2 = 100  # 第二阶段训练轮数
LEARNING_RATE_STAGE1 = 0.001  # 第一阶段学习率
LEARNING_RATE_STAGE2 = 0.0001  # 第二阶段学习率
NUM_CLASSES = 200  # AppClassNet 类别数
NUM_EXPERTS = 3  # MoE专家头数量
ROUTING_TYPE = 'hard'  # 路由类型: 'softmax' 或 'hard'
NUM_WORKERS = 8  # 数据加载的worker数量
PIN_MEMORY = True  # 确保启用pin_memory
PREFETCH_FACTOR = 8  # 增加预取因子

# 自动混合精度训练配置
USE_AMP = True  # 启��自动混合精度训练


# 查找并加载模型检查点
def load_checkpoint(checkpoint_path=None, model=None, optimizer=None, stage="stage1"):
    """加载模型检查点以继续训练"""
    start_epoch = 0
    best_val_acc = 0.0

    if checkpoint_path is None:
        # 寻找最新的epoch检查点
        checkpoint_files = glob.glob(f"{RESULTS_PATH}/param/{stage}_epoch*.pth")
        if checkpoint_files:
            # 提取轮次数并找到最大的
            epochs = [int(f.split('epoch')[-1].split('.')[0]) for f in checkpoint_files]
            max_epoch = max(epochs)
            checkpoint_path = f"{RESULTS_PATH}/param/{stage}_epoch{max_epoch}.pth"
            start_epoch = max_epoch  # 从下一个轮次开始
        else:
            # 如果没有epoch检查点，寻找最终模型检查点
            final_checkpoint = f"{RESULTS_PATH}/param/{stage}_final.pth"
            if os.path.isfile(final_checkpoint):
                checkpoint_path = final_checkpoint
            else:
                log_message(f"未找到{stage}阶段可用的检查点，将从头开始训练")
                return start_epoch, best_val_acc, model, optimizer

    # 检查文件是否存在
    if not os.path.isfile(checkpoint_path):
        log_message(f"检查点 {checkpoint_path} 不存在，将从头开始训练")
        return start_epoch, best_val_acc, model, optimizer

    # 加载检查点
    log_message(f"加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)

    # 检查检查点类型
    if isinstance(checkpoint, dict):
        # 完整检查点
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])

            # 处理优化器 - 第一阶段有多个优化器
            if stage == "stage1" and 'optimizers' in checkpoint and optimizer is not None:
                # 假设传入的optimizer是一个列表
                for i, opt_state in enumerate(checkpoint['optimizers']):
                    if i < len(optimizer):
                        optimizer[i].load_state_dict(opt_state)
            elif 'optimizer_state_dict' in checkpoint and optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1  # 从下一个epoch开始
            if 'val_accuracy' in checkpoint:
                best_val_acc = checkpoint['val_accuracy']
            elif 'best_val_acc' in checkpoint:
                best_val_acc = checkpoint['best_val_acc']
        else:
            # 仅模型参数
            model.load_state_dict(checkpoint)
    else:
        # 仅模型参数
        model.load_state_dict(checkpoint)

    log_message(
        f"成功加载第{stage}阶段检查点，从第 {start_epoch} 轮开始继续训练，当前最佳验证准确率: {best_val_acc:.2f}%")
    return start_epoch, best_val_acc, model, optimizer


# 加载预训练ResNet18模型并迁移参数到MoE模型
def load_pretrained_weights(moe_model, pretrained_path):
    """
    从预训练ResNet18模型加载权重到MoE模型的backbone部��
    
    Args:
        moe_model: MoE4Model实例
        pretrained_path: 预训练ResNet18模型的路径
    
    Returns:
        加载了预训练权重的MoE模型
    """
    if not os.path.exists(pretrained_path):
        log_message(f"预训练模型 {pretrained_path} 不存在，将使用随机初始化的权重")
        return moe_model

    log_message(f"从 {pretrained_path} 加载预训练ResNet18权重")

    # 加载预训练ResNet18模型
    pretrained_dict = torch.load(pretrained_path)

    # 检查加载的对象类型
    if isinstance(pretrained_dict, dict) and 'model_state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['model_state_dict']

    # 创建一个新字典，只包含backbone相关的参数
    backbone_dict = {}

    # 遍历预训练模型的所有参数
    for k, v in pretrained_dict.items():
        # 将ResNet参数映射到MoE backbone
        if k.startswith('conv1') or k.startswith('bn1') or k.startswith('layer'):
            # 添加backbone前缀以匹配MoE模型的键名
            backbone_key = f'backbone.{k}'
            backbone_dict[backbone_key] = v

    # 仅加载匹配的参数
    moe_state_dict = moe_model.state_dict()

    # 检查参数形状匹配
    for k, v in backbone_dict.items():
        if k in moe_state_dict and moe_state_dict[k].shape == v.shape:
            moe_state_dict[k] = v
        else:
            if k in moe_state_dict:
                log_message(f"形状不匹配: {k}, 预训练: {v.shape}, MoE: {moe_state_dict[k].shape}")
            else:
                log_message(f"MoE模型中不存在键: {k}")

    # 加载修改后的状态字典
    moe_model.load_state_dict(moe_state_dict)
    log_message("预训练权重成功加载到MoE模型backbone")

    return moe_model


# 第一阶段训练：单独训练每个专家
def train_stage1(model, train_loaders, val_loader, test_loader, device, resume_training=False):
    """
    第一阶段训练：分别训练每个专家
    """
    log_message(f"开始第一阶段训练...")
    writer = SummaryWriter(os.path.join(RESULTS_PATH, 'logs', 'stage1'))

    # 保存初始模型
    initial_state = model.state_dict()

    # 冻结backbone参数
    for param in model.backbone.parameters():
        param.requires_grad = False

    log_message("已冻结backbone参数，仅训练专家分类器")

    # 为每个专家创建优化器
    optimizers = []
    for i in range(len(model.experts)):
        # 只优化当前专家的参数
        optimizer = optim.Adam(model.experts[i].parameters(), lr=LEARNING_RATE_STAGE1)
        optimizers.append(optimizer)

    # 创建混合精度训练的scaler
    scaler = GradScaler() if USE_AMP else None

    # 添加恢复训练功能
    start_epoch = 0
    best_val_acc = 0.0
    if resume_training:
        start_epoch, best_val_acc, model, optimizers = load_checkpoint(None, model, optimizers, stage="stage1")

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, EPOCHS_STAGE1):
        epoch_start_time = time.time()
        log_message(f"第一阶段训练 - Epoch {epoch + 1}/{EPOCHS_STAGE1}")
        model.train()
        total_epoch_train_time = 0

        # 分别训练每个专家
        for expert_idx, (optimizer, train_loader) in enumerate(zip(optimizers, train_loaders)):
            expert_epoch_start_time = time.time()
            log_message(f"    训练专家 {expert_idx + 1}/{len(model.experts)}")
            start_class, end_class = model.class_ranges[expert_idx]
            epoch_loss = 0
            correct = 0
            total = 0
            expert_batch_train_time = 0

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
                expert_batch_train_time += batch_time_taken
                # 统计
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            # 记录每个专家的训练损失和准确率
            avg_loss = epoch_loss / len(train_loader)
            accuracy = correct / total
            expert_epoch_time_taken = time.time() - expert_epoch_start_time
            total_epoch_train_time += expert_epoch_time_taken
            writer.add_scalar(f'expert{expert_idx}/train_loss', avg_loss, epoch)
            writer.add_scalar(f'expert{expert_idx}/train_accuracy', accuracy, epoch)
            log_message(
                f"  \t专家{expert_idx} - 训练损失: {avg_loss:.4f}, 准确率: {accuracy:.4f}, 耗时: {expert_epoch_time_taken:.2f}s (Avg Batch: {expert_batch_train_time / len(train_loader):.2f}s)")

        epoch_time_taken = time.time() - epoch_start_time

        # 每个epoch结束后验证
        # 验证集评估
        valid_start_time = time.time()
        val_results = validate_experts(model, val_loader, criterion, writer, epoch, device, "val")
        valid_end_time = time.time()
        log_message(f"  验证集评估结果:")
        for expert_idx in range(len(model.experts)):
            if expert_idx in val_results['expert_accuracies']:
                log_message(f"    \t专家{expert_idx} - 验证损失: {val_results['expert_losses'][expert_idx]:.4f}, "
                            f"准确率: {val_results['expert_accuracies'][expert_idx]:.4f}")
        log_message(f"  \t验证集评估耗时: {valid_end_time - valid_start_time:.2f}s")

        # 测试集评估
        test_start_time = time.time()
        test_results = validate_experts(model, test_loader, criterion, writer, epoch, device, "test")
        test_end_time = time.time()
        log_message(f"  测试集评估结果:")
        for expert_idx in range(len(model.experts)):
            if expert_idx in test_results['expert_accuracies']:
                log_message(f"    \t专家{expert_idx} - 测试损失: {test_results['expert_losses'][expert_idx]:.4f}, "
                            f"准确率: {test_results['expert_accuracies'][expert_idx]:.4f}")
        log_message(f"  \t测试集评估耗时: {test_end_time - test_start_time:.2f}s")

        # 保存阶段性检查点
        if (epoch + 1) % 10 == 0 or (epoch + 1) == EPOCHS_STAGE1:
            checkpoint_path = os.path.join(RESULTS_PATH, 'param', f'stage1_epoch{epoch + 1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizers': [opt.state_dict() for opt in optimizers],
            }, checkpoint_path)
            log_message(f"已保存检查点到 {checkpoint_path}")

    # 保存最终的第一阶段模型
    final_checkpoint_path = os.path.join(RESULTS_PATH, 'param', 'stage1_final.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
    }, final_checkpoint_path)
    log_message(f"第一阶段训练完成，模型已保存到 {final_checkpoint_path}")
    writer.close()

    return model


def validate_experts(model, data_loader, criterion, writer, epoch, device, split="val"):
    """验证每个专家的性能"""
    model.eval()
    expert_losses = {i: 0 for i in range(len(model.experts))}
    expert_correct = {i: 0 for i in range(len(model.experts))}
    expert_total = {i: 0 for i in range(len(model.experts))}

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            features = model.backbone(inputs)

            # 对每个专家单独评估
            for expert_idx, expert in enumerate(model.experts):
                start_class, end_class = model.class_ranges[expert_idx]

                # 只评估该专家负责的类别范围内的样本
                mask = (targets >= start_class) & (targets <= end_class)

                if mask.sum() > 0:
                    expert_inputs = features[mask]
                    expert_targets = targets[mask] - start_class  # 调整目标标签

                    # 获取专家输出
                    outputs = expert(expert_inputs)

                    # 计算损失
                    loss = criterion(outputs, expert_targets)
                    expert_losses[expert_idx] += loss.item() * mask.sum().item()

                    # 计算准确率
                    _, predicted = outputs.max(1)
                    expert_total[expert_idx] += mask.sum().item()
                    expert_correct[expert_idx] += predicted.eq(expert_targets).sum().item()

    # 记录每个专家的验证损失和准确率
    expert_accuracies = {}
    for expert_idx in range(len(model.experts)):
        if expert_total[expert_idx] > 0:
            expert_losses[expert_idx] = expert_losses[expert_idx] / expert_total[expert_idx]
            expert_accuracies[expert_idx] = expert_correct[expert_idx] / expert_total[expert_idx]
            writer.add_scalar(f'expert{expert_idx}/{split}_loss', expert_losses[expert_idx], epoch)
            writer.add_scalar(f'expert{expert_idx}/{split}_accuracy', expert_accuracies[expert_idx], epoch)

    return {
        'expert_losses': expert_losses,
        'expert_accuracies': expert_accuracies,
        'expert_correct': expert_correct,
        'expert_total': expert_total
    }


# 第二阶段训练：仅训练路由器
def train_stage2(model, train_loader, val_loader, test_loader, device, resume_training=False):
    """
    第二阶段训练：冻结专家参数，只训练路由部分作为粗粒度分类器
    """
    log_message(f"开始第二阶段训练...")
    writer = SummaryWriter(os.path.join(RESULTS_PATH, 'logs', 'stage2'))

    # 冻结backbone和专家参数
    for param in model.backbone.parameters():
        param.requires_grad = False
    for expert in model.experts:
        for param in expert.parameters():
            param.requires_grad = False

    # 确保路由器参数可训练
    for param in model.router.parameters():
        param.requires_grad = True

    # 打印和确认参数训练状态
    log_message("参数训练状态检查:")
    for name, param in model.named_parameters():
        if 'router' in name:
            log_message(f"  {name}: requires_grad={param.requires_grad}")

    # 创建优化器，只优化路由器参数
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE_STAGE2)

    # 使用交叉熵损失函数，用于路由器分类任务
    criterion = nn.CrossEntropyLoss()

    # 添加恢复训练功能
    start_epoch = 0
    best_val_acc = 0.0
    if resume_training:
        start_epoch, best_val_acc, model, optimizer = load_checkpoint(None, model, optimizer, stage="stage2")

    # 创建混合精度训练的scaler
    scaler = GradScaler() if USE_AMP else None

    for epoch in range(start_epoch, EPOCHS_STAGE2):
        epoch_start_time = time.time()
        log_message(f"第二阶段训练 - Epoch {epoch + 1}/{EPOCHS_STAGE2}")
        model.train()  # 确保模型处于训练模式

        epoch_loss = 0
        correct = 0
        total = 0
        epoch_batch_train_time = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            batch_start_time = time.time()
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            # 将原始类别标签转换为对应的专家索引
            expert_targets = get_expert_indices(targets, model.class_ranges).to(device)

            optimizer.zero_grad(set_to_none=True)  # 更快地重置梯度

            # 使用混合精度训练
            if USE_AMP and scaler is not None:
                with autocast():
                    # 前向传播，只计算到路由器
                    features = model.backbone(inputs)
                    _, router_logits = model.router(features)

                    # 计算损失 - 路由器输出与专家目标之间的交叉熵
                    loss = criterion(router_logits, expert_targets)

                # 使用scaler进行反向传播和优化
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # 前向传播，只计算到路由器
                features = model.backbone(inputs)
                _, router_logits = model.router(features)

                # 计算损失 - 路由器输出与专家目标之间的交叉熵
                loss = criterion(router_logits, expert_targets)

                # 反向传播和优化
                loss.backward()
                optimizer.step()

            batch_time_taken = time.time() - batch_start_time
            epoch_batch_train_time += batch_time_taken

            # 统计路由准确率
            epoch_loss += loss.item()
            _, predicted_experts = router_logits.max(1)
            total += targets.size(0)
            correct += predicted_experts.eq(expert_targets).sum().item()

        # 记录训练损失和路由准确率
        avg_loss = epoch_loss / len(train_loader)
        routing_accuracy = correct / total
        epoch_time_taken = time.time() - epoch_start_time
        writer.add_scalar('train_loss', avg_loss, epoch)
        writer.add_scalar('routing_accuracy', routing_accuracy, epoch)
        log_message(
            f"  训练损失: {avg_loss:.4f}, 路由准确率: {routing_accuracy:.4f}, 耗时: {epoch_time_taken:.2f}s (Avg Batch: {epoch_batch_train_time / len(train_loader):.2f}s)")

        # 每轮结束后评估完整模型性能
        val_loss, val_accuracy, val_class_accuracies = validate_full_model(model, val_loader, nn.CrossEntropyLoss(),
                                                                           device, "valid")
        writer.add_scalar('val_accuracy', val_accuracy, epoch)

        test_loss, test_accuracy, test_class_accuracies = validate_full_model(model, test_loader, nn.CrossEntropyLoss(),
                                                                              device, "test")
        writer.add_scalar('test_accuracy', test_accuracy, epoch)

        # 记录每个类别区间的准确率
        for i, (start, end) in enumerate(model.class_ranges):
            writer.add_scalar(f'val_accuracy_classes_{start}-{end}', val_class_accuracies[i], epoch)
            writer.add_scalar(f'test_accuracy_classes_{start}-{end}', test_class_accuracies[i], epoch)

        # 保存检查点
        if (epoch + 1) % 5 == 0 or (epoch + 1) == EPOCHS_STAGE2:
            checkpoint_path = os.path.join(RESULTS_PATH, 'param', f'stage2_epoch{epoch + 1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'routing_accuracy': routing_accuracy,
                'val_accuracy': val_accuracy,
                'test_accuracy': test_accuracy,
            }, checkpoint_path)
            log_message(f"已保存检查点到 {checkpoint_path}")

    # 保存最终模型
    final_checkpoint_path = os.path.join(RESULTS_PATH, 'param', 'stage2_final.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'routing_accuracy': routing_accuracy,
        'val_accuracy': val_accuracy,
        'test_accuracy': test_accuracy,
    }, final_checkpoint_path)
    log_message(f"第二阶段训练完成，模型已保存到 {final_checkpoint_path}")
    writer.close()

    return model


# 辅助函数：将原始类别标签转换为对应的专家索引
def get_expert_indices(targets, class_ranges):
    """
    将原始类别标签映射到对应的专家索引
    
    Args:
        targets: 原始类别标签
        class_ranges: 专家负责的类别范围列表 [(start1, end1), (start2, end2), ...]
    
    Returns:
        专家索引张量
    """
    expert_indices = torch.zeros_like(targets)
    for expert_idx, (start_class, end_class) in enumerate(class_ranges):
        expert_indices[(targets >= start_class) & (targets <= end_class)] = expert_idx
    return expert_indices


def validate_full_model(model, data_loader, criterion, device, split="val"):
    """验证完整模型性能，并返回每个类别区间的准确率"""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    # 为每个类别区间跟踪准确率
    class_correct = [0] * len(model.class_ranges)
    class_total = [0] * len(model.class_ranges)

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            # 使用混合精度推理
            if USE_AMP:
                with autocast():
                    # 使用推理模式
                    outputs = model.inference(inputs)

                    # 计算损失
                    loss = criterion(outputs, targets)
            else:
                # 使用推理模式
                outputs = model.inference(inputs)

                # 计算损失
                loss = criterion(outputs, targets)

            val_loss += loss.item()

            # 计算准确率
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 计算每个类别区间的准确率
            for i, (start, end) in enumerate(model.class_ranges):
                mask = (targets >= start) & (targets <= end)
                if mask.sum() > 0:
                    class_total[i] += mask.sum().item()
                    class_correct[i] += predicted[mask].eq(targets[mask]).sum().item()

    # 计算平均损失和总体准确率
    avg_loss = val_loss / len(data_loader)
    accuracy = correct / total

    # 计算每个类别区间的准确率
    class_accuracies = [correct / total if total > 0 else 0 for correct, total in zip(class_correct, class_total)]

    log_message(f"\t{split.capitalize()} loss: {avg_loss:.4f}, accuracy: {accuracy:.4f}")
    for i, (start, end) in enumerate(model.class_ranges):
        log_message(
            f"\t  类别 {start}-{end} 准确率: {class_accuracies[i]:.4f} ({class_correct[i]}/{class_total[i]})")

    return avg_loss, accuracy, class_accuracies


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
    log_message(f"第一阶段学习率: {LEARNING_RATE_STAGE1}, 训练轮数: {EPOCHS_STAGE1}")
    log_message(f"第二阶段学习率: {LEARNING_RATE_STAGE2}, 训练轮数: {EPOCHS_STAGE2}")
    log_message(f"数据集路径: {DATASET_PATH}")
    log_message(f"结果保存路径: {RESULTS_PATH}")
    log_message(f"使用设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    log_message(f"自动混合精度: {USE_AMP}")
    log_message(f"工作进程数: {NUM_WORKERS}")
    log_message(f"专家数量: {NUM_EXPERTS}")
    log_message(f"路由类型: {ROUTING_TYPE}")
    log_message(f"预训练模型路径: {PRETRAINED_RESNET18_PATH}")

    # 配置数据并行训练，利用多个GPU
    multi_gpu = torch.cuda.device_count() > 1
    log_message(f"使用GPU数量: {torch.cuda.device_count()}")

    # 添加继续训练选项
    RESUME_STAGE1 = True  # 设置是否从检查点继续第一阶段训练
    RESUME_STAGE2 = True  # 设置是否从检查点继续第二阶段训练

    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    train_loaders, full_train_loader, val_loader, test_loader = get_dataloaders()

    # 从训练数据中获取输入形状
    sample_input, _ = next(iter(full_train_loader))
    input_channels = sample_input.shape[1]
    input_height = sample_input.shape[2]
    input_width = sample_input.shape[3]

    log_message(f"模型输入形状: 通道={input_channels}, 高度={input_height}, 宽度={input_width}")

    # 创建模型
    model = MoE4Model(
        total_classes=NUM_CLASSES,
        class_ranges=[(0, 99), (100, 149), (150, 199)],
        routing_type=ROUTING_TYPE,
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

    # 进行两阶段训练
    model = train_stage1(model, train_loaders, val_loader, test_loader, device, resume_training=RESUME_STAGE1)
    model = train_stage2(model, full_train_loader, val_loader, test_loader, device, resume_training=RESUME_STAGE2)

    log_message("训练完成！")
    log_message(f"=== 训练结束于 {datetime.datetime.now().strftime('%Y%m%d_%H%M%S')} ===")
