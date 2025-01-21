# 当代人工智能实验五：多模态情感分析

情感分析是自然语言处理中的一个重要任务，其目的是从文本中识别出情感倾向（如积极、消极或中性）。随着社交媒体的普及，用户在发布信息时往往会同时使用文本和图像来表达情感，这使得传统的基于纯文本的情感分析方法存在局限性。多模态情感分析通过结合文本和图像信息，能够更准确地捕捉用户的真实情感。

本次实验中，我设计了一种图片+文本的融合模型，考虑到后续的消融实验对比，添加了不同的运行选项（option=0为仅图像，option=1为仅文本，option=2为图像+文本的融合模型）。

## 代码结构

```
Contemporary-AI-lab5/
├── data/			# 数据集
│   ├── 1.jpg
│   ├── 1.txt
│   ├── 2.jpg
│   ├── 2.txt
│   └── ...
├── train.txt
├── test_without_label.txt
|
├── lab5_resnet18.ipynb		# kaggle上运行的结果
├── lab5_resnet50.ipynb		# kaggle上运行的结果
├── run_all.py		# 用于复现实验过程
|
├── requirements.txt	# kaggle中生成的项目依赖
├── README.md
├── config.py		# 配置文件，存储所有超参数和路径信息
├── models/
│   ├── __init__.py
│   ├── image_extractor.py	# 图像特征提取器模块
│   ├── text_extractor.py	# 文本特征提取器模块
│   ├── fusion_model.py		# 多模态融合模型模块
│   └── cross_modal_attention.py	# 交叉模态注意力模块
├── utils/
│   ├── __init__.py
│   ├── data_utils.py		# 数据处理模块
│   ├── train_utils.py		# 训练和验证模块
│   └── plot_utils.py		# 绘图模块
├── main.py					# 主程序模块
└── predict.py				# 预测模块
```

kaggle运行结果见`lab5_resnet18.ipynb`和`lab5_resnet50.ipynb`

## 运行步骤
1. 安装依赖库
   ```sh
   pip install -r requirements.txt

2. 运行主程序，进行消融实验，注意config中的设置

   ```sh
   python main.py
   ```

3. 利用保存的多模态模型，进行预测

   ```sh
   python predict.py
   ```

由于设备因素，本次实验在kaggle上运行，本地的依赖可能会有所冲突，在kaggle上只能以ipynb的形式运行，命令行参数的设置和模块封装较为麻烦，因此将各模块定义统一写在同一个运行文件中，参数在各个代码块中定义，详细运行结果请参照lab5_resnet18.ipynb和lab5_resnet50.ipynb中的内容，其中包含调参过程以及消融实验的对比结果。

若要在本地复现实验过程，请运行`run_all.py`，其中运行路径已从kaggle中的路径改为相对路径，并且在kaggle的代码基础上添加了各个部分详细的注释。

```
python run_all.py
```

为使得代码结构清晰，已将kaggle上运行的`lab5.ipynb`封装成各个模块，参考代码结构部分，可以考虑使用命令行参数替换配置文件的设置，进一步优化代码。