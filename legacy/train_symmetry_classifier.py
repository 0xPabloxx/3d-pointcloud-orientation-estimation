#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对称性分类网络训练脚本

支持：
- 3种网络架构的消融实验（GlobalConcat / PointConcat / NoUpright）
- 灵活的category选择（支持部分category训练）
- 类别权重平衡（处理不平衡数据）
- 详细的训练日志和评估

作者: Claude
创建日期: 2025-11-25
"""

import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import time

from dataloader_symmetry import get_symmetry_dataloaders
from models import (
    SymmetryClassifier_GlobalConcat,
    SymmetryClassifier_PointConcat,
    SymmetryClassifier_NoUpright
)


def get_model(model_name, num_classes=5):
    """根据名称获取模型"""
    models = {
        'global_concat': SymmetryClassifier_GlobalConcat,
        'point_concat': SymmetryClassifier_PointConcat,
        'no_upright': SymmetryClassifier_NoUpright
    }
    if model_name not in models:
        raise ValueError(f"未知模型: {model_name}. 可选: {list(models.keys())}")
    return models[model_name](num_classes=num_classes)


def compute_class_weights(dataloader, num_classes=5):
    """计算类别权重（用于处理不平衡数据）"""
    class_counts = np.zeros(num_classes)

    for batch in dataloader:
        labels = batch['label'].numpy()
        for label in labels:
            class_counts[label] += 1

    # 避免除零
    class_counts = np.maximum(class_counts, 1)

    # 计算权重：样本少的类别权重大
    total = class_counts.sum()
    weights = total / (num_classes * class_counts)

    return torch.FloatTensor(weights)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        points = batch['points'].to(device)
        upright_vec = batch['upright_vec'].to(device)
        labels = batch['label'].to(device)

        # Forward
        optimizer.zero_grad()
        logits = model(points, upright_vec)
        loss = criterion(logits, labels)

        # Backward
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total:.4f}'
        })

    avg_loss = total_loss / len(train_loader)
    avg_acc = correct / total

    return avg_loss, avg_acc


def evaluate(model, val_loader, criterion, device, num_classes=5):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    # 混淆矩阵
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)

    # Per-class统计
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating'):
            points = batch['points'].to(device)
            upright_vec = batch['upright_vec'].to(device)
            labels = batch['label'].to(device)

            logits = model(points, upright_vec)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # 更新混淆矩阵
            for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                confusion_matrix[t, p] += 1
                class_total[t] += 1
                if t == p:
                    class_correct[t] += 1

    avg_loss = total_loss / len(val_loader)
    avg_acc = correct / total

    # Per-class准确率
    class_acc = {}
    K_mapping = {0: -1, 1: 0, 2: 1, 3: 2, 4: 4}  # idx to K
    for i in range(num_classes):
        if class_total[i] > 0:
            K = K_mapping[i]
            class_acc[K] = class_correct[i] / class_total[i]
        else:
            class_acc[K_mapping[i]] = 0.0

    return avg_loss, avg_acc, confusion_matrix, class_acc


def save_checkpoint(model, optimizer, epoch, best_acc, save_path):
    """保存模型checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")


def load_checkpoint(model, optimizer, load_path, device):
    """加载模型checkpoint"""
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']
    print(f"Checkpoint loaded: {load_path} (Epoch {epoch}, Best Acc: {best_acc:.4f})")
    return epoch, best_acc


def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 保存实验配置
    config_path = save_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Config saved: {config_path}")

    # 加载数据
    print("\n" + "="*80)
    print("加载数据集")
    print("="*80)
    train_loader, val_loader, test_loader = get_symmetry_dataloaders(
        annotation_file=args.annotation_file,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_points=args.num_points
    )

    # 计算类别权重
    if args.use_class_weights:
        print("\n计算类别权重...")
        class_weights = compute_class_weights(train_loader, num_classes=5)
        print(f"类别权重: {class_weights}")
        class_weights = class_weights.to(device)
    else:
        class_weights = None

    # 创建模型
    print("\n" + "="*80)
    print(f"创建模型: {args.model}")
    print("="*80)
    model = get_model(args.model, num_classes=5)
    model = model.to(device)

    # 统计参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {num_params:,}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 学习率调度器
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    else:
        scheduler = None

    # 加载checkpoint（如果有）
    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        start_epoch, best_acc = load_checkpoint(model, optimizer, args.resume, device)

    # 训练日志
    log_file = save_dir / 'training.log'
    log_data = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'class_acc': []
    }

    # 训练循环
    print("\n" + "="*80)
    print("开始训练")
    print("="*80)
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 80)

        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # 验证
        val_loss, val_acc, confusion_matrix, class_acc = evaluate(
            model, val_loader, criterion, device, num_classes=5
        )
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Per-class Acc: {class_acc}")

        # 学习率调度
        if scheduler:
            scheduler.step()
            print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        # 保存日志
        log_data['train_loss'].append(train_loss)
        log_data['train_acc'].append(train_acc)
        log_data['val_loss'].append(val_loss)
        log_data['val_acc'].append(val_acc)
        log_data['class_acc'].append(class_acc)

        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_path = save_dir / 'best_model.pth'
            save_checkpoint(model, optimizer, epoch, best_acc, best_model_path)
            print(f"✅ New best model! Val Acc: {best_acc:.4f}")

        # 定期保存checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = save_dir / f'checkpoint_epoch_{epoch+1}.pth'
            save_checkpoint(model, optimizer, epoch, best_acc, checkpoint_path)

    # 最终测试
    print("\n" + "="*80)
    print("最终测试")
    print("="*80)

    # 加载最佳模型
    best_model_path = save_dir / 'best_model.pth'
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path)['model_state_dict'])

    test_loss, test_acc, confusion_matrix, class_acc = evaluate(
        model, test_loader, criterion, device, num_classes=5
    )

    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    print(f"Per-class Acc: {class_acc}")
    print("\nConfusion Matrix:")
    print(confusion_matrix)

    # 保存测试结果
    test_results = {
        'test_loss': test_loss,
        'test_acc': test_acc,
        'class_acc': class_acc,
        'confusion_matrix': confusion_matrix.tolist()
    }
    with open(save_dir / 'test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)

    print(f"\n✅ 训练完成! 最佳验证准确率: {best_acc:.4f}")
    print(f"结果保存在: {save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练对称性分类网络')

    # 数据相关
    parser.add_argument('--annotation_file', type=str,
                        default='data_annotation/symmetry_annotations.json',
                        help='标注文件路径')
    parser.add_argument('--data_dir', type=str,
                        default='data/full_mn40_normal_resampled_ply',
                        help='数据集目录')
    parser.add_argument('--num_points', type=int, default=10000,
                        help='采样点数')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')

    # 模型相关
    parser.add_argument('--model', type=str, default='global_concat',
                        choices=['global_concat', 'point_concat', 'no_upright'],
                        help='模型架构')

    # 训练相关
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--use_class_weights', action='store_true',
                        help='使用类别权重平衡')

    # 学习率调度
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'none'],
                        help='学习率调度器')
    parser.add_argument('--step_size', type=int, default=50,
                        help='StepLR的step_size')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='StepLR的gamma')

    # 保存相关
    parser.add_argument('--save_dir', type=str,
                        default='results/symmetry_classifier_exp1',
                        help='结果保存目录')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='保存checkpoint的频率（epoch）')
    parser.add_argument('--resume', type=str, default='',
                        help='恢复训练的checkpoint路径')

    args = parser.parse_args()

    # 打印配置
    print("="*80)
    print("训练配置")
    print("="*80)
    for key, value in vars(args).items():
        print(f"{key:20s}: {value}")
    print("="*80)

    main(args)
