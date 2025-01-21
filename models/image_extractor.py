import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

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
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
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