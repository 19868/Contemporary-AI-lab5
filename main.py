import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from models.fusion_model import FusionModel
from utils.data_utils import create_dataloader, get_valid_imagesPath_from_directory, get_texts_from_textsPath, text_preprocess
from utils.train_utils import train_model, validate_model, predict_model
from utils.plot_utils import plot_loss_curve, plot_accuracy_curve, plot_combined_loss_curves, plot_combined_accuracy_curves
from config import config
import pandas as pd
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

# 设置随机种子以保证结果可复现
torch.manual_seed(56)
np.random.seed(56)

# 数据准备
train_label_df = pd.read_csv(config["train_label_path"], sep=",")
column_dict = {"positive": 0, "negative": 1, "neutral": 2}
replace_func = np.vectorize(lambda x: column_dict.get(x, x))
new_df = train_label_df.copy()
new_df['tag'] = replace_func(new_df['tag'])
labels = list(new_df['tag'])

image_paths = get_valid_imagesPath_from_directory(config["folder_path"], new_df)
texts = get_texts_from_textsPath(config["folder_path"], new_df)

# 划分验证集
from sklearn.model_selection import train_test_split
image_paths_train, image_paths_val, texts_train, texts_val, labels_train, labels_val = train_test_split(
    image_paths, texts, labels, test_size=0.2, random_state=5)

# 文本预处理
tokenized_texts_train = text_preprocess(texts_train)
tokenized_texts_val = text_preprocess(texts_val)

# 图像数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 构建Dataset
from utils.data_utils import Dataset
dataset_train = Dataset(image_paths_train, tokenized_texts_train, labels_train, transform)
dataset_val = Dataset(image_paths_val, tokenized_texts_val, labels_val, transform)

# 创建数据加载器
train_loader, val_loader = create_dataloader(dataset_train, dataset_val, config["batch_size"])

# 模型训练和消融实验
criterion = nn.CrossEntropyLoss()
options = [0, 1, 2]
all_train_losses = []
all_val_losses = []
all_train_accs = []
all_val_accs = []
all_batch_losses = []  # 用于记录每个模型的所有 batch loss

for option in options:
    if option == 0:
        print("Start training only using image...")
    elif option == 1:
        print("Start training only using text...")
    elif option == 2:
        print("Start training using fusion model...")

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    all_batch_loss = []  # 用于记录当前模型的所有 batch loss
    best_acc = 0
    early_stopping_patience = config["early_stopping_patience"]
    epochs_no_improve = 0

    model = FusionModel(config["num_classes"], option).to(config["device"])
    optimizer = Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = StepLR(optimizer, step_size=2, gamma=0.1)

    for epoch in range(config["epochs"]):
        train_loss, train_acc, batch_losses = train_model(model, train_loader, criterion, optimizer, config["device"])
        val_loss, val_acc = validate_model(model, val_loader, criterion, config["device"])

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # 将当前 epoch 的 batch loss 追加到 all_batch_loss 中
        all_batch_loss.extend(batch_losses)

        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f"Option: {option}, Epoch {epoch + 1}/{config['epochs']}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Best Val Acc: {best_acc:.4f}")

        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

        scheduler.step()

    torch.save(model.state_dict(), f'multi_model_option_{option}.pt')
    all_train_losses.append(train_losses)
    all_val_losses.append(val_losses)
    all_train_accs.append(train_accs)
    all_val_accs.append(val_accs)

# 绘制所有模型的 batch loss 曲线
plt.figure(figsize=(10, 6))
for option, batch_loss in enumerate(all_batch_losses):
    plt.plot(batch_loss, label=f"Option {option}")
plt.xlabel("Batch Index")
plt.ylabel("Loss")
plt.title("Batch Losses for Different Models")
plt.legend()
plt.show()

# 绘制所有模型的验证准确率曲线
plt.figure(figsize=(10, 6))
for option, val_acc in enumerate(all_val_accs):
    plt.plot(val_acc, label=f"Option {option}")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy for Different Models")
plt.legend()
plt.show()

print("Training finished")