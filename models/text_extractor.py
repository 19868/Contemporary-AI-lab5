import torch.nn as nn
from transformers import BertModel

class TextFeatureExtractor(nn.Module):
    """
    文本特征提取器类，基于预训练的BERT模型。

    功能:
        - 使用BERT模型提取文本特征。
        - 对提取的特征进行归一化处理。
    """
    def __init__(self, pretrained_model):
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