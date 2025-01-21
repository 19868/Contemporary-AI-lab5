import torch
import torch.nn as nn
import numpy as np

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