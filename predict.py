import torch
import pandas as pd
import numpy as np
from models.fusion_model import FusionModel
from utils.data_utils import Dataset, get_valid_imagesPath_from_directory, get_texts_from_textsPath, text_preprocess
from config import config
from torchvision import transforms

# 加载测试数据
test_df = pd.read_csv(config["test_path"], sep=",")
test_df.iloc[:, -1] = 0
test_labels = np.array(test_df['tag'])

# 数据预处理
image_paths_test = get_valid_imagesPath_from_directory(config["folder_path"], test_df)
test_texts = get_texts_from_textsPath(config["folder_path"], test_df)
tokenized_texts_test = text_preprocess(test_texts)

# 图像数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 构建测试数据集和加载器
dataset_test = Dataset(image_paths_test, tokenized_texts_test, test_labels, transform)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=config["batch_size"], shuffle=False)

# 加载模型
model = FusionModel(config["num_classes"], option=2).to(config["device"])
model.load_state_dict(torch.load('multi_model_option_2.pt', map_location=config["device"]))
model.eval()

# 预测
predictions = []
with torch.no_grad():
    for images, input_ids, attention_mask, _ in test_loader:
        images = images.to(config["device"])
        input_ids = input_ids.to(config["device"])
        attention_mask = attention_mask.to(config["device"])
        outputs = model(images, input_ids, attention_mask)
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())

# 生成预测文件
column_dict_ = {0: "positive", 1: "negative", 2: "neutral"}
replace_func = np.vectorize(lambda x: column_dict_.get(x, x))
test_df['tag'] = replace_func(predictions)
test_df.to_csv(config["output_path"], sep=',', index=False)

print("Prediction finished")