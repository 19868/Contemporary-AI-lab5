import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torchvision.models import resnet50, ResNet50_Weights
import re
import matplotlib.pyplot as plt

# 设置随机种子以保证结果可复现
torch.manual_seed(56)
np.random.seed(56)

# 加载预训练的BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
pretrained_model = BertModel.from_pretrained("bert-base-multilingual-cased")

max_length = 50  # 输入的最大文本长度
num_classes = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
folder_path = "./data/"

# 图像数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),         # 将图像转换为 Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
])


# 读取文本文件，处理编码问题
def get_texts_from_textsPath(folder_path, df):
    """
    从指定路径读取文本文件内容。

    参数:
        folder_path (str): 包含文本文件的文件夹路径。
        df (pd.DataFrame): 包含文件名（guid）的DataFrame。

    返回:
        texts (list): 包含所有文本文件内容的列表。

    功能:
        遍历DataFrame中的每个文件名，尝试以GB18030编码读取对应的文本文件。
        如果文件无法解码，则跳过该文件并打印错误信息。
    """
    texts = []
    for ind in df['guid']:
        file = folder_path + str(ind) + ".txt"
        try:
            with open(file, "r", encoding="GB18030") as infile:
                content = infile.read()
        except UnicodeDecodeError:
            print(f"Failed to decode file: {file}")
            continue
        texts.append(content)
    return texts


# 清洗文本
def clean_text(text):
    """
    清洗文本，去除非字母和数字字符。

    参数:
        text (str): 输入的文本字符串。

    返回:
        cleaned_text (str): 清洗后的文本字符串。

    功能:
        使用正则表达式去除文本中的非字母和数字字符。
    """
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # 去除非字母和数字字符
    return text


# 文本预处理
def text_preprocess(texts):
    """
    对文本列表进行预处理，包括清洗和分词。

    参数:
        texts (list): 包含文本字符串的列表。

    返回:
        tokenized_texts (list): 包含分词后的Tensor的列表。

    功能:
        对每个文本进行清洗，然后使用BERT分词器进行分词和编码。
        每个文本被编码为一个固定长度的Tensor，填充或截断至`max_length`。
    """
    cleaned_texts = [clean_text(text) for text in texts]
    tokenized_texts = [
        tokenizer(text, padding='max_length', max_length=max_length, truncation=True, return_tensors="pt") for text in
        cleaned_texts]
    return tokenized_texts


# 图片和文本混合数据集
class Dataset(torch.utils.data.Dataset):
    """
    自定义数据集类，用于加载和处理图像和文本数据。

    功能:
        - 将图像路径、文本数据和标签封装为一个PyTorch数据集。
        - 支持图像的预处理和文本的编码。
    """
    def __init__(self, image_paths, tokenized_texts, labels, transform=None):
        """
        初始化数据集。

        参数:
            image_paths (list): 图像文件路径列表。
            tokenized_texts (list): 分词后的文本数据（Tensor）列表。
            labels (list): 数据标签列表。
            transform (callable, optional): 图像预处理函数，默认为None。
        """
        self.image_paths = image_paths
        self.transform = transform
        self.input_ids = [x['input_ids'].squeeze(0) for x in tokenized_texts]
        self.attention_mask = [x['attention_mask'].squeeze(0) for x in tokenized_texts]
        self.labels = labels

    def __getitem__(self, index):
        """
        获取数据集中的一个样本。

        参数:
            index (int): 样本索引。

        返回:
            tuple: 包含图像、文本输入ID、注意力掩码和标签的元组。
        """
        input_ids = self.input_ids[index]
        attention_mask = self.attention_mask[index]
        labels = torch.tensor(self.labels[index])
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        image = self.transform(image)
        return image, input_ids, attention_mask, labels

    def __len__(self):
        """
        获取数据集的样本数量。

        返回:
            int: 数据集的样本数量。
        """
        return len(self.input_ids)


# 依据train.txt获取训练集图片路径列表
def get_valid_imagesPath_from_directory(folder_path, df):
    """
    获取有效的图像路径列表。

    参数:
        folder_path (str): 包含图像文件的文件夹路径。
        df (pd.DataFrame): 包含文件名（guid）的DataFrame。

    返回:
        image_paths (list): 包含有效图像路径的列表。

    功能:
        遍历DataFrame中的每个文件名，尝试读取对应的图像文件。
        如果图像文件不存在或无效，则跳过该文件并打印错误信息。
    """
    image_paths = []
    for ind in df['guid']:
        image_path = folder_path + str(ind) + ".jpg"
        try:
            image = cv2.imread(image_path)
            height, width, channels = image.shape
            image_paths.append(image_path)
        except Exception as e:
            print(f"File '{image_path}' not found or invalid: {e}")
            continue
    return image_paths


# 数据准备
train_label_path = "train.txt"
train_label_df = pd.read_csv(train_label_path, sep=",")
column_dict = {"positive": 0, "negative": 1, "neutral": 2}
# 使用 np.vectorize 替换 'tag' 列的值
replace_func = np.vectorize(lambda x: column_dict.get(x, x))
new_df = train_label_df.copy()  # 创建副本以避免修改原始 DataFrame
new_df['tag'] = replace_func(new_df['tag'])
labels = list(new_df['tag'])

image_paths = get_valid_imagesPath_from_directory(folder_path, new_df)
texts = get_texts_from_textsPath(folder_path, new_df)

# 划分验证集
image_paths_train, image_paths_val, texts_train, texts_val, labels_train, labels_val = train_test_split(
    image_paths, texts, labels, test_size=0.2, random_state=5)

# 文本预处理
tokenized_texts_train = text_preprocess(texts_train)
tokenized_texts_val = text_preprocess(texts_val)

# 构建Dataset
dataset_train = Dataset(image_paths_train, tokenized_texts_train, labels_train, transform)
dataset_val = Dataset(image_paths_val, tokenized_texts_val, labels_val, transform)


# 图片特征提取模型定义
class ImageFeatureExtractor(nn.Module):
    """
    图像特征提取器类，基于预训练的ResNet-50模型。

    功能:
        - 使用ResNet-50提取图像特征。
        - 对提取的特征进行归一化处理。
    """
    def __init__(self):
        """
        初始化图像特征提取器。

        使用预训练的ResNet-50模型，并冻结其权重。
        """
        super(ImageFeatureExtractor, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)  # 使用预训练的ResNet-50作为图片特征提取器
        self.norm = nn.LayerNorm(1000)  # 归一化层

    def forward(self, image):
        """
        前向传播函数，提取图像特征。

        参数:
            image (Tensor): 输入图像张量。

        返回:
            Tensor: 提取的图像特征。
        """
        features = self.resnet(image)
        return features


# 文本特征提取模型定义
class TextFeatureExtractor(nn.Module):
    """
    文本特征提取器类，基于预训练的BERT模型。

    功能:
        - 使用BERT模型提取文本特征。
        - 对提取的特征进行归一化处理。
    """
    def __init__(self):
        """
        初始化文本特征提取器。

        使用预训练的BERT模型，并冻结其权重。
        """
        super(TextFeatureExtractor, self).__init__()
        self.bert = pretrained_model
        for param in self.bert.parameters():
            param.requires_grad = False  # 冻结 BERT 权重
        self.norm = nn.LayerNorm(768)  # 归一化层

    def forward(self, input_ids, attention_mask):
        """
        前向传播函数，提取文本特征。

        参数:
            input_ids (Tensor): 文本输入ID张量。
            attention_mask (Tensor): 注意力掩码张量。

        返回:
            Tensor: 提取的文本特征。
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # 获取池化后的特征 [batch_size, 768]
        return pooled_output


class CrossModalAttention(nn.Module):
    """
    交叉模态注意力模块，用于融合图像和文本特征。

    功能:
        - 计算图像特征和文本特征之间的注意力权重。
        - 使用注意力权重加权文本特征。
    """
    def __init__(self, image_dim, text_dim, hidden_dim):
        """
        初始化交叉模态注意力模块。

        参数:
            image_dim (int): 图像特征的维度。
            text_dim (int): 文本特征的维度。
            hidden_dim (int): 隐藏层维度。
        """
        super(CrossModalAttention, self).__init__()
        self.query = nn.Linear(image_dim, hidden_dim)
        self.key = nn.Linear(text_dim, hidden_dim)
        self.value = nn.Linear(text_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, image_features, text_features):
        """
        前向传播函数，计算交叉模态注意力。

        参数:
            image_features (Tensor): 图像特征张量。
            text_features (Tensor): 文本特征张量。

        返回:
            Tensor: 加权后的文本特征。
        """
        Q = self.query(image_features)  # 图像特征的查询向量
        K = self.key(text_features)  # 文本特征的键向量
        V = self.value(text_features)  # 文本特征的值向量

        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.T) / np.sqrt(K.shape[-1])
        attention_weights = self.softmax(attention_scores)

        # 加权求和
        attended_text_features = torch.matmul(attention_weights, V)
        return attended_text_features


# 多模态融合模型定义
class FusionModel(nn.Module):
    """
    多模态融合模型类，支持图像、文本和融合特征的分类。

    功能:
        - 提取图像和文本特征。
        - 使用交叉模态注意力模块融合特征。
        - 提供三种模式：仅图像、仅文本和融合特征，可以用作消融实验对比。
    """
    def __init__(self, num_classes, option, hidden_dim=256):
        """
        初始化多模态融合模型。

        参数:
            num_classes (int): 分类任务的类别数。
            option (int): 模型选项（0: 仅图像，1: 仅文本，2: 融合特征）。
            hidden_dim (int, optional): 隐藏层维度，默认为256。
        """
        super(FusionModel, self).__init__()
        self.image_extractor = ImageFeatureExtractor()
        self.text_encoder = TextFeatureExtractor()
        self.option = option

        # 交叉模态注意力模块
        self.cross_modal_attention = CrossModalAttention(image_dim=1000, text_dim=768, hidden_dim=hidden_dim)

        # 分类器
        self.classifier0 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1000, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, num_classes)
        )

        self.classifier1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(768, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, num_classes)
        )

        self.classifier2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1000 + 768 + 256, hidden_dim * 2),  # 融合后的特征维度
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim * 2, num_classes)
        )

    def forward(self, image, input_ids, attention_mask):
        """
        前向传播函数，根据选项选择特征提取和分类方式。

        参数:
            image (Tensor): 输入图像张量。
            input_ids (Tensor): 文本输入ID张量。
            attention_mask (Tensor): 注意力掩码张量。

        返回:
            Tensor: 分类结果。
        """
        if self.option == 0:
            image_features = self.image_extractor(image)
            output = self.classifier0(image_features)
        elif self.option == 1:
            text_features = self.text_encoder(input_ids, attention_mask)
            output = self.classifier1(text_features)
        else:
            image_features = self.image_extractor(image)
            text_features = self.text_encoder(input_ids, attention_mask)

            # 使用交叉模态注意力
            attended_text_features = self.cross_modal_attention(image_features, text_features)

            # 融合特征
            fusion_features = torch.cat((image_features, text_features, attended_text_features), dim=-1)
            output = self.classifier2(fusion_features)
        return output


# 训练过程
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

    功能:
        对训练数据进行前向传播，计算损失，执行反向传播更新模型参数。
        记录每个batch的损失和整体准确率。
    """
    model.train()
    running_loss = 0
    total_correct = 0
    batch_losses = []  # 用于存储每个 batch 的损失值

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

        # 将当前 batch 的损失值存储到列表中
        batch_losses.append(loss.item())

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = total_correct.item() / len(train_loader.dataset)

    return epoch_loss, epoch_acc, batch_losses


# 验证过程
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

    功能:
        对验证数据进行前向传播，计算损失和准确率。
        不执行反向传播，不更新模型参数。
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


# 预测过程
def predict_model(model, test_loader, device):
    """
    使用模型进行预测。

    参数:
        model (nn.Module): 模型实例。
        test_loader (DataLoader): 测试数据加载器。
        device (torch.device): 测试设备（CPU或GPU）。

    返回:
        predictions (list): 模型的预测结果列表。

    功能:
        对测试数据进行前向传播，获取模型的预测结果。
        不执行反向传播，不更新模型参数。
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


# 为每个学习率绘制loss曲线
def plot_loss_curve(train_losses, val_losses, lr):
    """
    绘制训练和验证损失曲线。

    参数:
        train_losses (list): 训练损失列表。
        val_losses (list): 验证损失列表。
        lr (float): 学习率。

    功能:
        使用matplotlib绘制训练和验证损失曲线。
        保存为文件`loss_curve_lr_{lr}.png`。
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label=f'Train Loss (LR: {lr})')
    plt.plot(val_losses, label=f'Validation Loss (LR: {lr})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss Curves (LR: {lr})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'loss_curve_lr_{lr}.png')  # 保存为不同文件
    plt.show()


def plot_accuracy_curve(train_accs, val_accs, lr):
    """
    绘制训练和验证准确率曲线。

    参数:
        train_accs (list): 训练准确率列表。
        val_accs (list): 验证准确率列表。
        lr (float): 学习率。

    功能:
        使用matplotlib绘制训练和验证准确率曲线。
        保存为文件`accuracy_curve_lr_{lr}.png`。
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_accs, label=f'Train Accuracy (LR: {lr})')
    plt.plot(val_accs, label=f'Validation Accuracy (LR: {lr})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Training and Validation Accuracy Curves (LR: {lr})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'accuracy_curve_lr_{lr}.png')  # 保存为不同文件
    plt.show()


# 定义统一的绘图函数，对比不同的lr
def plot_combined_loss_curves(all_train_losses, all_val_losses, lrs):
    """
    绘制所有模型的训练和验证损失曲线。

    参数:
        all_train_losses (list): 所有模型的训练损失列表。
        all_val_losses (list): 所有模型的验证损失列表。
        lrs (list): 对应的学习率列表。

    功能:
        使用matplotlib绘制所有模型的训练和验证损失曲线。
        保存为文件`combined_loss_curve.png`。
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
    plt.savefig('combined_loss_curve.png')  # 保存为统一文件
    plt.show()


# 定义统一的准确率绘图函数
def plot_combined_accuracy_curves(all_train_accs, all_val_accs, lrs):
    """
    绘制所有模型的训练和验证准确率曲线。

    参数:
        all_train_accs (list): 所有模型的训练准确率列表。
        all_val_accs (list): 所有模型的验证准确率列表。
        lrs (list): 对应的学习率列表。

    功能:
        使用matplotlib绘制所有模型的训练和验证准确率曲线。
        保存为文件`combined_accuracy_curve.png`。
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
    plt.savefig('combined_accuracy_curve.png')  # 保存为统一文件
    plt.show()

# 不同学习率的模型训练和验证
torch.cuda.set_device(0)
criterion = nn.CrossEntropyLoss()
lrs = [1e-3, 1e-4, 1e-5]
batch_size = 64
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
option = 2


if option == 0:
    print("Start training only using image...")
elif option == 1:
    print("Start training only using text...")
elif option == 2:
    print("Start training using fusion model...")


# 用于存储所有学习率的损失和准确率
all_train_losses = []
all_val_losses = []
all_train_accs = []
all_val_accs = []
all_batch_losses_list = []  # 用于存储每个学习率的 all_batch_losses

for lr in lrs:
    # 用于记录训练和验证的损失和准确率
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    all_batch_losses = []  # 用于记录所有 epoch 的连续的 batch_losses
    best_acc = 0
    
    model = FusionModel(num_classes, option)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    num_epochs = 6
    
    for epoch in range(num_epochs):
        train_loss, train_acc, batch_losses= train_model(model, loader_train, criterion, optimizer, device)
        val_loss, val_acc = validate_model(model, loader_val, criterion, device)

        # 记录损失和准确率
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # 将当前 epoch 的 batch_losses 追加到 all_batch_losses 中
        all_batch_losses.extend(batch_losses)  # 使用 extend 将 batch_losses 追加到连续列表中
        
        if val_acc > best_acc:
            best_acc = val_acc
            # torch.save(model.state_dict(), 'multi_model.pt')

        print(f"Batch size: {batch_size}, LR: {lr}, Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Best Val Acc: {best_acc:.4f}")
        scheduler.step()
        
    # 调用绘图函数
    plot_loss_curve(train_losses, val_losses, lr)
    plot_accuracy_curve(train_accs, val_accs, lr)
    # 绘制所有 epoch 的 batch_losses 连续变化
    plt.plot(all_batch_losses, label=f"LR: {lr}")
    plt.xlabel("Batch Index")
    plt.ylabel("Loss")
    plt.title("Continuous Batch Losses During Training")
    plt.legend()
    plt.show()

    # 将当前学习率的 all_batch_losses 添加到全局列表中
    all_batch_losses_list.append(all_batch_losses)
    
    # 将当前学习率的损失和准确率存储起来
    all_train_losses.append(train_losses)
    all_val_losses.append(val_losses)
    all_train_accs.append(train_accs)
    all_val_accs.append(val_accs)


# 在循环结束后调用统一绘图函数
plot_combined_loss_curves(all_train_losses, all_val_losses, lrs)
plot_combined_accuracy_curves(all_train_accs, all_val_accs, lrs)

for lr, all_batch_losses in zip(lrs, all_batch_losses_list):
    plt.plot(all_batch_losses, label=f"LR: {lr}")
plt.xlabel("Batch Index")
plt.ylabel("Loss")
plt.title("Continuous Batch Losses for Different Learning Rates")
plt.legend()
plt.show()

print("Training finished")

# 设置设备
torch.cuda.set_device(0)
criterion = nn.CrossEntropyLoss()
lr = 1e-4
batch_size = 64
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
options = [0, 1, 2]

# 用于存储所有模型的损失和准确率
all_train_losses = []
all_val_losses = []
all_train_accs = []
all_val_accs = []
all_batch_losses = []  # 用于记录每个模型的所有 batch loss

# 消融实验
for option in options:
    if option == 0:
        print("Start training only using image...")
    elif option == 1:
        print("Start training only using text...")
    elif option == 2:
        print("Start training using fusion model...")

    # 用于记录训练和验证的损失和准确率
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    all_batch_loss = []  # 用于记录当前模型的所有 batch loss
    best_acc = 0
    early_stopping_patience = 5
    epochs_no_improve = 0

    model = FusionModel(num_classes, option)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    num_epochs = 20

    for epoch in range(num_epochs):
        train_loss, train_acc, batch_losses = train_model(model, loader_train, criterion, optimizer, device)
        val_loss, val_acc = validate_model(model, loader_val, criterion, device)

        # 记录损失和准确率
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

        print(f"Option: {option}, Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Best Val Acc: {best_acc:.4f}")

        # 检查是否触发早停
        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

        scheduler.step()

    # 保存模型状态
    torch.save(model.state_dict(), f'multi_model_option_{option}.pt')

    # 将当前模型的损失和准确率存储到全局列表中
    all_train_losses.append(train_losses)
    all_val_losses.append(val_losses)
    all_train_accs.append(train_accs)
    all_val_accs.append(val_accs)
    all_batch_losses.append(all_batch_loss)

print("Training finished")

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

print("Start predicting...")

# 读取test文件
test_path = "test_without_label.txt"
test_df = pd.read_csv(test_path, sep=",")
test_df.iloc[:, -1] = 0
test_labels = np.array(test_df['tag'])

# tests数据处理并构建数据加载器
image_paths_test = get_valid_imagesPath_from_directory(folder_path, test_df)
test_texts = get_texts_from_textsPath(folder_path, test_df)

tokenized_texts_test = text_preprocess(test_texts)
dataset_test = Dataset(image_paths_test, tokenized_texts_test, test_labels, transform)
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

# 加载模型权重
state_dict = torch.load('multi_model_option_2.pt', map_location=device, weights_only=True)

# 创建模型实例
best_model = FusionModel(num_classes, option).to(device)

# 加载权重
best_model.load_state_dict(state_dict)

test_predictions = predict_model(best_model, loader_test, device)
test_predictions = np.array(test_predictions)

# 生成预测文件
column_dict_ = {0: "positive", 1: "negative", 2: "neutral"}

# 使用 np.vectorize 替换情感标签
replace_func = np.vectorize(lambda x: column_dict_.get(x, x))
test_df['tag'] = replace_func(test_predictions)

# 保存预测结果
test_df.to_csv('predict.txt', sep=',', index=False)

print("Prediction finished")