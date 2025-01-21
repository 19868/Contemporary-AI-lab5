import torch
import pandas as pd
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import re
from transformers import BertTokenizer

# 加载预训练的BERT分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

def get_texts_from_textsPath(folder_path, df):
    """
    从指定路径读取文本文件内容。

    参数:
        folder_path (str): 包含文本文件的文件夹路径。
        df (pd.DataFrame): 包含文件名（guid）的DataFrame。

    返回:
        texts (list): 包含所有文本文件内容的列表。
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

def clean_text(text):
    """
    清洗文本，去除非字母和数字字符。

    参数:
        text (str): 输入的文本字符串。

    返回:
        cleaned_text (str): 清洗后的文本字符串。
    """
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # 去除非字母和数字字符
    return text

def text_preprocess(texts):
    """
    对文本列表进行预处理，包括清洗和分词。

    参数:
        texts (list): 包含文本字符串的列表。

    返回:
        tokenized_texts (list): 包含分词后的Tensor的列表。
    """
    cleaned_texts = [clean_text(text) for text in texts]
    tokenized_texts = [
        tokenizer(text, padding='max_length', max_length=50, truncation=True, return_tensors="pt") for text in cleaned_texts]
    return tokenized_texts

def get_valid_imagesPath_from_directory(folder_path, df):
    """
    获取有效的图像路径列表。

    参数:
        folder_path (str): 包含图像文件的文件夹路径。
        df (pd.DataFrame): 包含文件名（guid）的DataFrame。

    返回:
        image_paths (list): 包含有效图像路径的列表。
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

class Dataset(Dataset):
    """
    自定义数据集类，用于加载和处理图像和文本数据。
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

def create_dataloader(train_data, val_data, batch_size):
    """
    创建数据加载器。

    参数:
        train_data (Dataset): 训练数据集。
        val_data (Dataset): 验证数据集。
        batch_size (int): 批量大小。

    返回:
        train_loader (DataLoader): 训练数据加载器。
        val_loader (DataLoader): 验证数据加载器。
    """
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader