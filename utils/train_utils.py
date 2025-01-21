import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def train_model(model, train_loader, criterion, optimizer, device):
    """
    训练模型的一个epoch。

    参数:
        model (nn.Module): 模型实例。
        train_loader (DataLoader): 训练数据加载器。
        criterion (nn.Module): 损失函数。
        optimizer (torch.optim.Optimizer): 优化器。
        device (torch.device): 训练设备（CPU或GPU）。

    返回:
        epoch_loss (float): 该epoch的平均训练损失。
        epoch_acc (float): 该epoch的平均训练准确率。
        batch_losses (list): 每个batch的损失列表。
    """
    model.train()
    running_loss = 0
    total_correct = 0
    batch_losses = []

    for batch_idx, (images, input_ids, attention_mask, labels) in enumerate(train_loader):
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, input_ids, attention_mask)
        _, preds = torch.max(outputs, 1)
        total_correct += torch.sum(preds == labels)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        batch_losses.append(loss.item())

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = total_correct.item() / len(train_loader.dataset)
    return epoch_loss, epoch_acc, batch_losses

def validate_model(model, val_loader, criterion, device):
    """
    验证模型的性能。

    参数:
        model (nn.Module): 模型实例。
        val_loader (DataLoader): 验证数据加载器。
        criterion (nn.Module): 损失函数。
        device (torch.device): 验证设备（CPU或GPU）。

    返回:
        val_loss (float): 验证集的平均损失。
        val_acc (float): 验证集的平均准确率。
    """
    model.eval()
    running_loss = 0
    total_correct = 0

    with torch.no_grad():
        for images, input_ids, attention_mask, labels in val_loader:
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total_correct += torch.sum(preds == labels)

    val_loss = running_loss / len(val_loader)
    val_acc = total_correct.item() / len(val_loader.dataset)
    return val_loss, val_acc

def predict_model(model, test_loader, device):
    """
    使用模型进行预测。

    参数:
        model (nn.Module): 模型实例。
        test_loader (DataLoader): 测试数据加载器。
        device (torch.device): 测试设备（CPU或GPU）。

    返回:
        predictions (list): 模型的预测结果列表。
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for images, input_ids, attention_mask, _ in test_loader:
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(images, input_ids, attention_mask)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
    return predictions