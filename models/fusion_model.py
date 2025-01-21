from models.image_extractor import ImageFeatureExtractor
from models.text_extractor import TextFeatureExtractor
from models.cross_modal_attention import CrossModalAttention
import torch
import torch.nn as nn
from transformers import BertModel

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
        self.text_encoder = TextFeatureExtractor(pretrained_model=BertModel.from_pretrained("bert-base-multilingual-cased"))
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