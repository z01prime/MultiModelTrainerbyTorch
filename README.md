# MultiModelTrainerbyTorch

本项目提供了一个使用[PlantVillage(with augmentation)](https://data.mendeley.com/datasets/tywbtsjrjv/1) 数据集的训练框架，支持在图像分类任务中使用多种预训练模型进行训练。脚本支持训练多个模型，自动保存每个模型的最佳表现，并绘制多模型训练的对比图 。

## 环境和硬件平台 (Environment and Hardware Platform)

### 软件环境 (Software Environment)
以下是代码运行所需的软件版本：
- **操作系统 (OS):** win-64
- **Python 版本 (Python Version):** Python 3.9+
- **依赖库 (Dependencies):**
  - **PyTorch:** `torch==2.5.1+cu121`
  - **TorchAudio:** `torchaudio==2.5.1+cu121`
  - **TorchVision:** `torchvision==0.20.1+cu121`
  - **NumPy:** `numpy==1.26.3`
  - **Pandas:** `pandas==2.2.3`
  - **Matplotlib:** `matplotlib==3.9.2`
  - **OpenCV:** `opencv-python==4.10.0.84`
  - **Pillow:** `pillow==10.2.0`
  - **TQDM:** `tqdm==4.67.0`
  - **Scikit-learn:** `scikit-learn==1.5.2`

可使用以下命令安装所有依赖：
 ```bash
 pip install -r requirements.txt
 ```

### 硬件平台 (Hardware Platform)
- **GPU:** NVIDIA GeForce RTX 3050 (4 GB 显存) 

### GPU 支持 (GPU Support)
- 本代码支持 GPU 加速，训练时会自动检测并使用 GPU。如果系统中没有可用 GPU，将回退到 CPU。
- **CUDA 支持版本:** CUDA 12.1  

支持以下模型的训练：

1. **ResNet18** (`resnet18`)  
2. **ResNet50** (`resnet50`)  
3. **MobileNetV2** (`mobilenet_v2`)  
4. **EfficientNet-B0** (`efficientnet_b0`)  
5. **DenseNet121** (`densenet121`)  

如果需要支持更多的模型，可以在 `initialize_model` 函数中按照以下格式添加对应模型的初始化逻辑：

```python
elif model_name == '新模型名':
    model = 新模型模块(pretrained=True)
    model.classifier 或 model.fc = nn.Linear(输入特征数, num_classes)
```

添加后，可以通过命令行参数指定新模型进行训练。

## 数据集分割 (data_split)

### 介绍

此部分代码实现了数据集的自动分割，将原始数据集划分为训练集（train）和验证集（val）,将指定路径下的数据集按照设定的比例划分为训练集和验证集，方便进行后续的模型训练和评估。

### 功能

+ 自动读取数据集目录，获取每个类别（文件夹）中的图片。
+ 将每个类别的数据按照指定比例划分为训练集和验证集。
+ 为训练集和验证集创建相应的子文件夹，并将图片移动到对应的文件夹中。
+ 生成划分统计信息，并保存为 CSV 文件。

### 使用说明

1. **准备数据集**  
   你的数据集应该遵循如下结构：

```plain
dataset_path/
    class_1/
        image1.jpg
        image2.jpg
        ...
    class_2/
        image1.jpg
        image2.jpg
        ...
    ...
```

每个子文件夹代表一个类别，文件夹内包含该类别的所有图片。

2. **配置参数**  
   在代码中，你可以设置以下参数：
    - `dataset_path`：数据集所在的目录路径。
    - `test_frac`：测试集所占比例，默认为 0.2（即 20% 的数据将作为验证集）。
    - `random_seed`：随机种子，用于确保划分过程的可复现性，默认为 123。
3. **运行代码**  
   你可以通过直接运行脚本来划分数据集。示例如下：

```bash
python data_split.py --dataset_path ./leaf_diseases --test_frac 0.2 --random_seed 42
```

4. **结果**  
   运行完成后，数据集将被划分为两个子文件夹：`train` 和 `val`，分别存放训练集和验证集的数据。数据集将被按类别组织在 `train` 和 `val` 文件夹内，最终的文件结构如下所示：

```plain
dataset_path_split/
    train/
        class_1/
            image1.jpg
            image2.jpg
            ...
        class_2/
            image1.jpg
            image2.jpg
            ...
        ...
    val/
        class_1/
            image1.jpg
            image2.jpg
            ...
        class_2/
            image1.jpg
            image2.jpg
            ...
        ...
```

5. **统计信息**  
   数据划分的统计结果将保存在 `数据量统计.csv` 文件中，内容包括每个类别在训练集和验证集中的数据量。示例如下：

```plain
class,trainset,testset,total
class_1,100,25,125
class_2,120,30,150
...
```

## 训练模型(train)

### 介绍

该脚本用于在给定的数据集上训练多个深度学习模型，并评估其性能。支持使用不同的预训练模型（如 ResNet、MobileNet 等）进行训练，并在训练过程中保存最佳模型和最终模型。

在运行之前，数据集应具有以下结构：

```plain
dataset_dir/
    ├── train/
    │   ├── class_1/
    │   ├── class_2/
    │   └── ...
    ├── val/
    │   ├── class_1/
    │   ├── class_2/
    │   └── ...
```

+ `train/` 目录：包含训练数据，按类别组织。
+ `val/` 目录：包含验证数据，按类别组织。

### 代码结构

+ `configure_environment`：配置训练环境（如图像显示、警告设置等）。
+ `get_transforms`：定义训练和测试的数据预处理流程。
+ `load_data`：加载训练集和验证集，并返回数据加载器。
+ `initialize_model`：根据模型名称初始化相应的预训练模型。
+ `train_epoch`：执行一个训练轮次，计算损失和准确率。
+ `evaluate_model`：评估模型在验证集上的表现。
+ `plot_results`：绘制多模型的训练结果对比图。
+ `main`：主程序，控制训练流程，支持多模型训练。

### 使用说明

运行此脚本时，您可以通过命令行指定以下参数：

+ `--dataset_dir`：数据集所在目录的路径，**必填**。该目录需要包含 `train` 和 `val` 子目录，分别用于训练集和验证集。
+ `--models`：需要训练的模型列表，**可选**，默认使用 `resnet18`。支持的模型包括 `resnet18`, `resnet50`, `mobilenet_v2`, `efficientnet_b0`, `densenet121` 等。
+ `--batch_size`：训练时的 batch 大小，默认为 `16`。
+ `--epochs`：训练的轮数，默认为 `5`。

命令行示例

```bash
python train.py --dataset_dir ./leaf_diseases --models resnet18 resnet50 --batch_size 64 --epochs 10
```

上面的命令将使用 `resnet18` 和 `resnet50` 两个模型在 `./leaf_diseases` 数据集上训练，批量大小为 `64`，训练 `10` 轮。



脚本在每一轮训练时会保存当前最优的模型（即验证集准确率最高的模型），并将其保存在 `checkpoints/` 目录下。每个模型保存的文件名包括模型名称和轮数（例如 `resnet18-10.pth`）。此外，最后一轮训练结束后，模型也会保存为 `checkpoints/{model_name}-{epochs}-last.pth`。

+ 最优模型：`checkpoints/{model_name}-best.pth`
+ 最后轮模型：`checkpoints/{model_name}-{epochs}-last.pth`

训练结束后，脚本会绘制训练和验证损失、训练和验证准确率的变化曲线，并将结果保存为 `multi_model_training_results.jpg` 图片文件。此图包含多个模型的对比，便于评估不同模型的训练效果。

## 图像分类（image_classifier）

这是一个基于 PyTorch 的图像分类器，可以加载预训练的模型对单张图片进行分类预测。通过终端传递参数来指定图像、模型路径等信息，支持对预测结果进行可视化（图像与柱状图）和保存。

### 功能

+ 加载预训练的深度学习模型（如 ResNet、VGG 等）。
+ 对输入的图像进行分类预测。
+ 显示并保存预测的图像与对应的柱状图。
+ 输出分类结果，并将预测结果保存为 CSV 文件。

### 文件结构

```plain
.
├── image_classifier.py          # 图像分类脚本
├── checkpoints/                 # 存放训练好的模型
│   └── resnet18-20.pth         # 示例模型文件
├── idx_to_labels.npy            # 类别索引到标签的映射文件
└── test_img/                    # 测试图片文件夹
    └── test.jpg                 # 示例测试图片
```

### 使用方法

通过命令行运行图像分类

1. 在终端中使用 `python` 命令运行 `image_classifier.py` 脚本，并传递以下参数：
   - `--image_path`：待分类图片的路径。
   - `--model`：模型检查点文件的路径（必需）。
   - `--labels`：类别索引到标签的映射文件路径（默认为 `'idx_to_labels.npy'`）。
   - `--font`：字体文件路径（默认为 `'SimHei.ttf'`），用于在图片上标注分类结果。
   - `--no-save`：是否禁用保存预测结果（默认为 `False`，添加此参数后不会保存结果）。

例如：

```bash
python image_classifier.py --model .\checkpoints\resnet18_5-best.pth --labels idx_to_labels.npy --image_path .\test_img\test.jpg --no-save --font SimHei.ttf
```

输出结果

+ 分类预测结果：图像将被分类，并在图像上方标注预测的类别及其对应的置信度。
+ 预测图与柱状图：生成包含分类结果的柱状图，并保存为 `output/预测图+柱状图.jpg`。
+ 预测结果 CSV 文件：预测结果将保存在 `output/predictions.csv` 文件中，包含每个类别和其对应的置信度。

## 实时分类(camera_classfier)

### 介绍

该脚本通过加载训练的深度学习模型，处理摄像头实时捕获的视频帧，预测目标类别，并显示预测结果及其置信度。同时，该系统会计算帧率（FPS），以便监控性能。

### 功能

+ 实时检测并显示摄像头输入的视频帧。
+ 使用预训练模型进行目标分类。
+ 显示 Top-5 预测类别及其置信度。
+ 支持设备选择（`cpu` 或 `cuda`）。
+ 可通过命令行参数自定义输入选项。

### 使用方法

命令行参数说明

脚本支持以下命令行参数：

| 参数                   | 描述                                                         | 默认值  |
| ---------------------- | ------------------------------------------------------------ | ------- |
| `--model_path`         | 预训练模型文件路径（必填）。                                 | 无      |
| `--idx_to_labels_path` | `idx_to_labels.npy` 文件路径，用于映射类别索引到标签（必填）。 | 无      |
| `--camera_id`          | 摄像头设备 ID（如 `0` 为系统默认摄像头）。                   | `0`     |
| `--font_path`          | 字体文件路径（如 `'SimHei.ttf'`）。                          | `None`  |
| `--device`             | 推理设备类型（`cpu` 或 `cuda`）。                            | `'cpu'` |


示例命令

以下示例展示了如何使用脚本：

```bash
python .\camera_classifier.py --model_path .\checkpoints\resnet18_5-best.pth --idx_to_labels_path ./idx_to_labels.npy --camera_id 0 --font_path ./SimHei.ttf --device cuda
```

脚本工作流程：

1. **初始化**：
   - 加载预训练模型。
   - 加载类别映射文件（`idx_to_labels.npy`）。
   - 初始化字体设置，用于显示预测结果。
2. **帧处理**：
   - 从摄像头捕获帧。
   - 将帧转换为 RGB 格式，进行预处理，并输入模型进行推理。
   - 获取 Top-5 预测结果，并在帧上叠加显示。
3. **显示**：
   - 在窗口中实时显示处理后的帧。
   - 按下 `q` 或 `ESC` 键退出程序。



