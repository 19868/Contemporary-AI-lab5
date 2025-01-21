import torch

config = {
    "max_length": 50,  # 输入的最大文本长度
    "num_classes": 3,  # 分类任务的类别数
    "learning_rate": 1e-4,  # 学习率
    "weight_decay": 1e-5,  # 权重衰减
    "batch_size": 64,  # 批量大小
    "epochs": 20,  # 训练轮数
    "early_stopping_patience": 5,  # 早停机制的耐心轮数
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # 使用的设备
    "folder_path": "./data/",  # 数据文件夹路径
    "train_label_path": "train.txt",  # 训练标签文件路径
    "test_path": "test_without_label.txt",  # 测试文件路径
    "output_path": "predict.txt",  # 预测结果保存路径
}