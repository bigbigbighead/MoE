# 查找并加载模型检查点
import glob
import os
from data_loading_mine import log_message


def load_checkpoint(RESULTS_PATH, checkpoint_path=None, model=None, optimizer=None, stage="stage1"):
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
    从预训练ResNet18模型加载权重到MoE模型的backbone部分

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

    # ResNet18中需要排除的层（通常是最终全连接分类层）
    exclude_layers = ['fc']

    # 遍历预训练模型的所有参数
    for k, v in pretrained_dict.items():
        # 排除分类层
        if not any(exclude_layer in k for exclude_layer in exclude_layers):
            # 添加backbone前缀以匹配MoE模型的键名
            backbone_key = f'backbone.{k}'
            backbone_dict[backbone_key] = v

    log_message(f"从预训练模型中提取了 {len(backbone_dict)} 个参数")

    # 检查MoE模型backbone中是否包含所有需要的键
    moe_state_dict = moe_model.state_dict()
    missing_keys = [k for k in backbone_dict.keys() if k not in moe_state_dict]
    unexpected_keys = [k for k in backbone_dict.keys() if
                       k in moe_state_dict and backbone_dict[k].shape != moe_state_dict[k].shape]

    if missing_keys:
        log_message(f"MoE模型中缺少以下键: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")

    if unexpected_keys:
        log_message(f"形状不匹配的键: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")

    # 仅加载匹配的参数
    matched_keys = 0
    for k, v in backbone_dict.items():
        if k in moe_state_dict and moe_state_dict[k].shape == v.shape:
            moe_state_dict[k] = v
            matched_keys += 1

    # 加载修改后的状态字典
    moe_model.load_state_dict(moe_state_dict)
    log_message(f"成功将 {matched_keys} 个预训练权重参数加载到MoE模型backbone（总共 {len(backbone_dict)} 个参数）")

    return moe_model
