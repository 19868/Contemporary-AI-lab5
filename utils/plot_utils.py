import matplotlib.pyplot as plt

def plot_loss_curve(train_losses, val_losses, lr):
    """
    绘制训练和验证损失曲线。

    参数:
        train_losses (list): 训练损失列表。
        val_losses (list): 验证损失列表。
        lr (float): 学习率。
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label=f'Train Loss (LR: {lr})')
    plt.plot(val_losses, label=f'Validation Loss (LR: {lr})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss Curves (LR: {lr})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'loss_curve_lr_{lr}.png')
    plt.show()

def plot_accuracy_curve(train_accs, val_accs, lr):
    """
    绘制训练和验证准确率曲线。

    参数:
        train_accs (list): 训练准确率列表。
        val_accs (list): 验证准确率列表。
        lr (float): 学习率。
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_accs, label=f'Train Accuracy (LR: {lr})')
    plt.plot(val_accs, label=f'Validation Accuracy (LR: {lr})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Training and Validation Accuracy Curves (LR: {lr})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'accuracy_curve_lr_{lr}.png')
    plt.show()

def plot_combined_loss_curves(all_train_losses, all_val_losses, lrs):
    """
    绘制所有模型的训练和验证损失曲线。

    参数:
        all_train_losses (list): 所有模型的训练损失列表。
        all_val_losses (list): 所有模型的验证损失列表。
        lrs (list): 对应的学习率列表。
    """
    plt.figure(figsize=(10, 5))
    for lr, train_losses, val_losses in zip(lrs, all_train_losses, all_val_losses):
        plt.plot(train_losses, label=f'Train Loss (LR: {lr})')
        plt.plot(val_losses, label=f'Validation Loss (LR: {lr})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Combined Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('combined_loss_curve.png')
    plt.show()

def plot_combined_accuracy_curves(all_train_accs, all_val_accs, lrs):
    """
    绘制所有模型的训练和验证准确率曲线。

    参数:
        all_train_accs (list): 所有模型的训练准确率列表。
        all_val_accs (list): 所有模型的验证准确率列表。
        lrs (list): 对应的学习率列表。
    """
    plt.figure(figsize=(10, 5))
    for lr, train_accs, val_accs in zip(lrs, all_train_accs, all_val_accs):
        plt.plot(train_accs, label=f'Train Accuracy (LR: {lr})')
        plt.plot(val_accs, label=f'Validation Accuracy (LR: {lr})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Combined Training and Validation Accuracy Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('combined_accuracy_curve.png')
    plt.show()