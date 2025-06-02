import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from utils.data_loading_mine import get_dataloaders, CLASS_RANGES, NUM_CLASSES, USE_AMP, log_message, RESULTS_PATH
from models.MoE_full import MixtureOfExperts


def train_coarse_classifier(model, optimizer, criterion, full_train_loader, val_loader, device, 
                           num_epochs=50, patience=5, scheduler=None, use_amp=USE_AMP):
    """
    训练粗分类器
    
    Args:
        model: MixtureOfExperts模型
        optimizer: 优化器
        criterion: 损失函数
        full_train_loader: 完整训练集加载器
        val_loader: 验证集加载器
        device: 计算设备
        num_epochs: 训练轮数
        patience: 早停耐心值
        scheduler: 学习率调度器
        use_amp: 是否使用混合精度训练
    """
    model.train()
    
    # 早停相关变量
    best_val_acc = 0.0
    epochs_no_improve = 0
    
    # 混合精度训练
    scaler = GradScaler() if use_amp else None
    
    # 记录训练过程
    train_losses = []
    val_accs = []
    
    log_message("开始训练粗分类器...")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        # 训练循环
        for inputs, targets in tqdm(full_train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 获取粗分类器的目标标签（专家索引）
            expert_targets = model.get_expert_target(targets)
            
            optimizer.zero_grad()
            
            if use_amp:
                with autocast():
                    # 只获取粗分类器输出
                    outputs = model(inputs, mode='coarse_only')
                    loss = criterion(outputs, expert_targets)
                
                # 混合精度训练的反向传播
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # 常规训练
                outputs = model(inputs, mode='coarse_only')
                loss = criterion(outputs, expert_targets)
                loss.backward()
                optimizer.step()
            
            # 记录精度和损失
            epoch_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(expert_targets).sum().item()
        
        if scheduler:
            scheduler.step()
            
        # 计算训练损失和准确率
        train_loss = epoch_loss / len(full_train_loader.dataset)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        
        # 在验证集上评估
        val_acc = validate_coarse_classifier(model, val_loader, device, use_amp)
        val_accs.append(val_acc)
        
        log_message(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        # 早停检查
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            # 保存最佳模型
            torch.save(model.state_dict(), f"{RESULTS_PATH}/param/coarse_classifier_best.pth")
            log_message(f"保存新的最佳模型，验证准确率: {val_acc:.2f}%")
        else:
            epochs_no_improve += 1
            log_message(f"验证准确率未提升 {epochs_no_improve}/{patience} epochs")
            
        if epochs_no_improve == patience:
            log_message(f"早停！验证准确率 {patience} epochs 未提升")
            break
    
    # 可视化训练过程
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Coarse Classifier Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accs)
    plt.title('Coarse Classifier Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_PATH}/coarse_classifier_training.png")
    
    return best_val_acc


def validate_coarse_classifier(model, val_loader, device, use_amp=USE_AMP):
    """验证粗分类器性能"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            expert_targets = model.get_expert_target(targets)
            
            if use_amp:
                with autocast():
                    outputs = model(inputs, mode='coarse_only')
            else:
                outputs = model(inputs, mode='coarse_only')
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(expert_targets).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy


def train_expert(model, expert_idx, optimizer, criterion, train_loader, val_loader, device, 
                num_epochs=100, patience=10, scheduler=None, use_amp=USE_AMP):
    """
    训练单个专家网络
    
    Args:
        model: MixtureOfExperts模型
        expert_idx: 专家索引
        optimizer: 优化器
        criterion: 损失函数
        train_loader: 专家训练数据加载器
        val_loader: 专家验证数据加载器
        device: 计算设备
        num_epochs: 训练轮数
        patience: 早停耐心值
        scheduler: 学习率调度器
        use_amp: 是否使用混合精度训练
    """
    # 早停相关变量
    best_val_acc = 0.0
    epochs_no_improve = 0
    
    # 混合精度训练
    scaler = GradScaler() if use_amp else None
    
    # 记录训练过程
    train_losses = []
    val_accs = []
    
    log_message(f"开始训练专家 {expert_idx}...")
    
    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        # 训练循环
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            if use_amp:
                with autocast():
                    # 对于共享骨干网络，我们需要单独训练每个专家头
                    if model.shared_backbone:
                        features = model.backbone(inputs)
                        outputs = model.expert_heads[expert_idx](features)
                    else:
                        outputs = model.experts[expert_idx](inputs)
                    
                    loss = criterion(outputs, targets)
                
                # 混合精度训练的反向传播
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # 常规训练
                if model.shared_backbone:
                    features = model.backbone(inputs)
                    outputs = model.expert_heads[expert_idx](features)
                else:
                    outputs = model.experts[expert_idx](inputs)
                
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            # 记录精度和损失
            epoch_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        if scheduler:
            scheduler.step()
            
        # 计算训练损失和准确率
        train_loss = epoch_loss / len(train_loader.dataset)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        
        # 在验证集上评估
        val_acc = validate_expert(model, expert_idx, val_loader, device, use_amp)
        val_accs.append(val_acc)
        
        log_message(f"Expert {expert_idx} - Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        # 早停检查
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            # 保存最佳模型
            torch.save(model.state_dict(), f"{RESULTS_PATH}/param/expert_{expert_idx}_best.pth")
            log_message(f"保存专家 {expert_idx} 的新最佳模型，验证准确率: {val_acc:.2f}%")
        else:
            epochs_no_improve += 1
            log_message(f"专家 {expert_idx} 验证准确率未提升 {epochs_no_improve}/{patience} epochs")
            
        if epochs_no_improve == patience:
            log_message(f"专家 {expert_idx} 早停！验证准确率 {patience} epochs 未提升")
            break
    
    # 可视化训练过程
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title(f'Expert {expert_idx} Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accs)
    plt.title(f'Expert {expert_idx} Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_PATH}/expert_{expert_idx}_training.png")
    
    return best_val_acc


def validate_expert(model, expert_idx, val_loader, device, use_amp=USE_AMP):
    """验证专家网络性能"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            if use_amp:
                with autocast():
                    if model.shared_backbone:
                        features = model.backbone(inputs)
                        outputs = model.expert_heads[expert_idx](features)
                    else:
                        outputs = model.experts[expert_idx](inputs)
            else:
                if model.shared_backbone:
                    features = model.backbone(inputs)
                    outputs = model.expert_heads[expert_idx](features)
                else:
                    outputs = model.experts[expert_idx](inputs)
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy


def finetune_moe(model, optimizer, criterion, full_train_loader, val_loader, device, 
                num_epochs=20, use_amp=USE_AMP, alpha=0.5, beta=0.5):
    """
    端到端微调混合专家模型
    
    Args:
        model: MixtureOfExperts模型
        optimizer: 优化器
        criterion: 损失函数
        full_train_loader: 完整训练数据加载器
        val_loader: 验证数据加载器
        device: 计算设备
        num_epochs: 训练轮数
        use_amp: 是否使用混合精度训练
        alpha: 粗分类器损失权重
        beta: 专家网络损失权重
    """
    # 加载之前训练的最佳模型
    log_message("加载先前训练的最佳模型参数")
    model.load_state_dict(torch.load(f"{RESULTS_PATH}/param/coarse_classifier_best.pth"))
    
    for expert_idx in range(len(CLASS_RANGES)):
        expert_state = torch.load(f"{RESULTS_PATH}/param/expert_{expert_idx}_best.pth")
        model.load_state_dict(expert_state, strict=False)
    
    # 混合精度训练
    scaler = GradScaler() if use_amp else None
    
    # 记录训练过程
    train_losses = []
    val_accs = []
    
    log_message("开始端到端微调混合专家模型...")
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        # 训练循环
        for inputs, targets in tqdm(full_train_loader, desc=f"Finetune Epoch {epoch+1}/{num_epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 获取粗分类器的目标标签（专家索引）
            expert_targets = model.get_expert_target(targets)
            
            optimizer.zero_grad()
            
            if use_amp:
                with autocast():
                    # 完整的前向传播
                    coarse_output, expert_id, local_preds, global_preds = model(inputs)
                    
                    # 粗分类器损失
                    coarse_loss = criterion(coarse_output, expert_targets)
                    
                    # 专家网络损失（仅考虑选定的专家）
                    batch_size = inputs.size(0)
                    expert_loss = torch.tensor(0.0).to(device)
                    
                    for i in range(batch_size):
                        e_id = expert_id[i].item()
                        local_target = targets[i].item() - CLASS_RANGES[e_id][0]
                        
                        if model.shared_backbone:
                            features = model.backbone(inputs[i:i+1])
                            expert_output = model.expert_heads[e_id](features)
                        else:
                            expert_output = model.experts[e_id](inputs[i:i+1])
                        
                        expert_loss += criterion(expert_output, torch.tensor([local_target]).to(device))
                    
                    expert_loss = expert_loss / batch_size
                    
                    # 总损失
                    loss = alpha * coarse_loss + beta * expert_loss
                
                # 混合精度训练的反向传播
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # 完整的前向传播
                coarse_output, expert_id, local_preds, global_preds = model(inputs)
                
                # 粗分类器损失
                coarse_loss = criterion(coarse_output, expert_targets)
                
                # 专家网络损失（仅考虑选定的专家）
                batch_size = inputs.size(0)
                expert_loss = torch.tensor(0.0).to(device)
                
                for i in range(batch_size):
                    e_id = expert_id[i].item()
                    local_target = targets[i].item() - CLASS_RANGES[e_id][0]
                    
                    if model.shared_backbone:
                        features = model.backbone(inputs[i:i+1])
                        expert_output = model.expert_heads[e_id](features)
                    else:
                        expert_output = model.experts[e_id](inputs[i:i+1])
                    
                    expert_loss += criterion(expert_output, torch.tensor([local_target]).to(device))
                
                expert_loss = expert_loss / batch_size
                
                # 总损失
                loss = alpha * coarse_loss + beta * expert_loss
                loss.backward()
                optimizer.step()
            
            # 记录精度和损失
            epoch_loss += loss.item() * inputs.size(0)
            total += targets.size(0)
            correct += global_preds.eq(targets.float()).sum().item()
        
        # 计算训练损失和准确率
        train_loss = epoch_loss / len(full_train_loader.dataset)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        
        # 在验证集上评估
        val_acc = evaluate_moe(model, val_loader, device, use_amp)
        val_accs.append(val_acc)
        
        log_message(f"Finetune Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{RESULTS_PATH}/param/moe_full_best.pth")
            log_message(f"保存微调后的最佳模型，验证准确率: {val_acc:.2f}%")
    
    # 可视化训练过程
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('End-to-end Finetuning Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accs)
    plt.title('End-to-end Finetuning Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_PATH}/finetuning.png")
    
    return best_val_acc


def evaluate_moe(model, data_loader, device, use_amp=USE_AMP):
    """评估完整的混合专家模型"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            if use_amp:
                with autocast():
                    _, _, _, global_preds = model(inputs)
            else:
                _, _, _, global_preds = model(inputs)
            
            total += targets.size(0)
            correct += global_preds.eq(targets.float()).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy


def test_moe(model, test_loader, device, use_amp=USE_AMP):
    """测试混合专家模型"""
    model.eval()
    
    all_targets = []
    all_predictions = []
    expert_counts = [0] * len(CLASS_RANGES)
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="测试中"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            if use_amp:
                with autocast():
                    coarse_output, expert_id, _, global_preds = model(inputs)
            else:
                coarse_output, expert_id, _, global_preds = model(inputs)
            
            # 统计每个专家被选择的次数
            for e_id in expert_id:
                expert_counts[e_id.item()] += 1
            
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(global_preds.cpu().numpy())
    
    # 计算评估指标
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='macro')
    recall = recall_score(all_targets, all_predictions, average='macro')
    f1 = f1_score(all_targets, all_predictions, average='macro')
    
    # 输出评估结果
    log_message("\n----- 测试结果 -----")
    log_message(f"准确率: {accuracy*100:.2f}%")
    log_message(f"精确率 (Macro): {precision*100:.2f}%")
    log_message(f"召回率 (Macro): {recall*100:.2f}%")
    log_message(f"F1分数 (Macro): {f1*100:.2f}%")
    
    # 输出专家分配情况
    total_samples = sum(expert_counts)
    log_message("\n----- 专家分配统计 -----")
    for i, count in enumerate(expert_counts):
        percentage = count / total_samples * 100
        log_message(f"专家 {i}: {count} 样本 ({percentage:.2f}%)")
    
    # 绘制混淆矩阵（可选，对于大量类别可能不直观）
    if NUM_CLASSES <= 30:  # 仅当类别数量较少时绘制混淆矩阵
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(all_targets, all_predictions)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f"{RESULTS_PATH}/confusion_matrix.png")
    
    return accuracy, precision, recall, f1


def main():
    # 设置随机种子以便结果可重复
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    
    # 检查CUDA是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_message(f"使用设备: {device}")
    
    # 获取数据加载器
    train_loaders, val_loaders, test_loaders, full_train_loader, val_loader, test_loader = get_dataloaders(CLASS_RANGES)
    
    # 检查数据形状
    for x, y in full_train_loader:
        log_message(f"输入数据形状: {x.shape}")
        input_channels = x.shape[1]
        input_height = x.shape[2] if len(x.shape) > 3 else 1
        input_width = x.shape[3] if len(x.shape) > 3 else x.shape[2]
        break
    
    # 创建混合专家模型
    model = MixtureOfExperts(
        class_ranges=CLASS_RANGES,
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        dropout_rate=0.2,
        shared_backbone=True  # 使用共享骨干网络
    )
    model = model.to(device)
    
    # 统计模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    log_message(f"模型总参数量: {total_params:,}")
    
    # 阶段1: 训练粗分类器
    log_message("\n===== 阶段1: 训练粗分类器 =====")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    # 如果使用共享骨干网络，需要冻结专家头部分
    if model.shared_backbone:
        for expert_head in model.expert_heads:
            for param in expert_head.parameters():
                param.requires_grad = False
    else:
        for expert in model.experts:
            for param in expert.parameters():
                param.requires_grad = False
    
    coarse_val_acc = train_coarse_classifier(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        full_train_loader=full_train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=50,
        patience=5,
        scheduler=scheduler,
        use_amp=USE_AMP
    )
    
    log_message(f"\n粗分类器训练完成，最佳验证准确率: {coarse_val_acc:.2f}%")
    
    # 阶段2: 训练专家网络
    log_message("\n===== 阶段2: 训练专家网络 =====")
    
    # 加载先前训练的最佳粗分类器
    model.load_state_dict(torch.load(f"{RESULTS_PATH}/param/coarse_classifier_best.pth"))
    
    # 如果使用共享骨干网络，需要冻结骨干网络和粗分类器
    if model.shared_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.coarse_head.parameters():
            param.requires_grad = False
    else:
        for param in model.coarse_classifier.parameters():
            param.requires_grad = False
    
    # 训练每个专家
    expert_val_accs = []
    for expert_idx in range(len(CLASS_RANGES)):
        log_message(f"\n----- 训练专家 {expert_idx} -----")
        
        # 解冻当前专家网络
        if model.shared_backbone:
            for param in model.expert_heads[expert_idx].parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.expert_heads[expert_idx].parameters(), lr=0.001)
        else:
            for param in model.experts[expert_idx].parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.experts[expert_idx].parameters(), lr=0.001)
        
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        
        expert_val_acc = train_expert(
            model=model,
            expert_idx=expert_idx,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loaders[expert_idx],
            val_loader=val_loaders[expert_idx],
            device=device,
            num_epochs=100,
            patience=10,
            scheduler=scheduler,
            use_amp=USE_AMP
        )
        
        expert_val_accs.append(expert_val_acc)
        
        # 再次冻结当前专家网络
        if model.shared_backbone:
            for param in model.expert_heads[expert_idx].parameters():
                param.requires_grad = False
        else:
            for param in model.experts[expert_idx].parameters():
                param.requires_grad = False
    
    log_message("\n所有专家网络训练完成")
    for i, acc in enumerate(expert_val_accs):
        log_message(f"专家 {i} 最佳验证准确率: {acc:.2f}%")
    
    # 阶段3: 端到端微调（可选）
    log_message("\n===== 阶段3: 端到端微调 =====")
    
    # 解冻所有参数
    for param in model.parameters():
        param.requires_grad = True
    
    # 使用较小的学习率进行微调
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    finetune_val_acc = finetune_moe(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        full_train_loader=full_train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=20,
        use_amp=USE_AMP,
        alpha=0.5,
        beta=0.5
    )
    
    log_message(f"\n端到端微调完成，最佳验证准确率: {finetune_val_acc:.2f}%")
    
    # 最终测试
    log_message("\n===== 最终测试 =====")
    model.load_state_dict(torch.load(f"{RESULTS_PATH}/param/moe_full_best.pth"))
    test_accuracy, test_precision, test_recall, test_f1 = test_moe(
        model=model,
        test_loader=test_loader,
        device=device,
        use_amp=USE_AMP
    )
    
    log_message("\n训练和评估完成！")


if __name__ == "__main__":
    main()
