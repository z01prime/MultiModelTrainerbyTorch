# coding: utf-8
import os
import warnings
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd


# --- 配置和环境初始化 ---
def configure_environment():
    """初始化环境配置。"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    warnings.filterwarnings("ignore")
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# --- 数据集和预处理 ---
def get_transforms():
    """定义训练和测试集的数据预处理流程。"""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4635, 0.4889, 0.4080], [0.1755, 0.1484, 0.1919])
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.4635, 0.4889, 0.4080], [0.1755, 0.1484, 0.1919])
    ])
    return train_transform, test_transform


def load_data(dataset_dir, train_transform, test_transform, batch_size):
    """加载数据集并返回数据加载器。"""
    train_path = os.path.join(dataset_dir, 'train')
    test_path = os.path.join(dataset_dir, 'val')

    train_dataset = datasets.ImageFolder(train_path, train_transform)
    test_dataset = datasets.ImageFolder(test_path, test_transform)

    idx_to_labels = {y: x for x, y in train_dataset.class_to_idx.items()}
    np.save('idx_to_labels.npy', idx_to_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader, len(train_dataset.classes)


# --- 模型初始化 ---
def initialize_model(model_name, num_classes, device):
    """根据模型名称初始化对应模型。"""
    print(device)
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'efficientnet_b0':
        from torchvision.models import efficientnet_b0
        model = efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'densenet121':
        from torchvision.models import densenet121
        model = densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError(f"不支持模型: {model_name}")

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个 epoch，并显示进度条。"""
    model.train()
    epoch_loss, correct, total = 0.0, 0, 0
    num_batches = len(train_loader)

    # 初始化进度条
    print(f"\n这一轮的进度:")

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

        # 显示进度条
        progress = (i + 1) / num_batches  # 计算当前进度
        num_stars = int(progress * 50)  # 进度条的长度为50个字符
        print(f"[{'*' * num_stars}{' ' * (50 - num_stars)}] {100 * progress:.2f}%", end="\r")

    accuracy = 100 * correct / total
    print()  # 换行
    return epoch_loss / num_batches, accuracy


def evaluate_model(model, test_loader, criterion, device):
    """评估模型性能。"""
    print("正在评估模型性能")
    model.eval()
    epoch_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            epoch_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    accuracy = 100 * correct / total
    return epoch_loss / len(test_loader), accuracy


def plot_results(results):
    """绘制多模型训练结果对比图。"""
    plt.figure(figsize=(15, 20))

    for model_name, stats in results.items():
        epochs = range(len(stats['train_losses']))

        # 训练损失
        plt.subplot(4, 1, 1)
        plt.plot(epochs, stats['train_losses'], label=f'{model_name} Train Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)

        # 测试损失
        plt.subplot(4, 1, 2)
        plt.plot(epochs, stats['test_losses'], label=f'{model_name} Test Loss')
        plt.title('Testing Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)

        # 训练准确率
        plt.subplot(4, 1, 3)
        plt.plot(epochs, stats['train_accuracies'], label=f'{model_name} Train Accuracy')
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)

        # 测试准确率
        plt.subplot(4, 1, 4)
        plt.plot(epochs, stats['test_accuracies'], label=f'{model_name} Test Accuracy')
        plt.title('Testing Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)

    plt.legend()
    plt.tight_layout()
    plt.savefig('multi_model_training_results.jpg')
    plt.show()


def save_results_to_csv(results, output_path='training_results.csv'):
    """
    将训练和测试的结果保存为 CSV 文件。

    :param results: dict，包含每个模型的训练和测试结果。
    :param output_path: str，CSV 文件保存路径。
    """
    data = []
    for model_name, stats in results.items():
        for epoch in range(len(stats['train_losses'])):
            data.append({
                'Model': model_name,
                'Epoch': epoch + 1,
                'Train Loss': stats['train_losses'][epoch],
                'Train Accuracy (%)': stats['train_accuracies'][epoch],
                'Test Loss': stats['test_losses'][epoch],
                'Test Accuracy (%)': stats['test_accuracies'][epoch],
            })

    # 转换为 DataFrame 并保存
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f'训练和测试结果已保存至 {output_path}')


# --- 主程序 ---
def main(args):
    device = configure_environment()
    train_transform, test_transform = get_transforms()
    train_loader, test_loader, num_classes = load_data(args.dataset_dir, train_transform, test_transform,
                                                       args.batch_size)

    results = {}
    for model_name in args.models:
        print(f"\n正在训练模型: {model_name}")

        model, optimizer, criterion = initialize_model(model_name, num_classes, device)

        train_losses, train_accuracies, test_losses, test_accuracies = [], [], [], []

        best_accuracy = 0.0  # 用于记录最佳验证集准确率
        # best_loss = float('inf')  # 或用于记录最低验证集损失

        for epoch in tqdm(range(args.epochs), desc=f'{model_name} 进程'):
            train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
            test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)

            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

            print(f"Epoch {epoch + 1}/{args.epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
                  f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

            # 保存最佳模型
            if test_accuracy > best_accuracy:  # 以验证准确率为准
                best_accuracy = test_accuracy
                os.makedirs('checkpoints', exist_ok=True)
                torch.save(model, f'checkpoints/{model_name}_{args.epochs}-best.pth')
                print(f'当前模型表现最佳，已保存至 checkpoints/{model_name}_{args.epochs}-best.pth')

        # 保存模型
        os.makedirs('checkpoints', exist_ok=True)
        # torch.save(model.state_dict(), f'checkpoints/{model_name}-{args.epochs}.pth')
        torch.save(model, f'checkpoints/{model_name}-{args.epochs}-last.pth')
        print(f'保存{model_name}模型最后一轮训练后的结果在checkpoints/{model_name}-{args.epochs}-last.pth中')
        # 记录结果
        results[model_name] = {'train_accuracies': train_accuracies, 'train_losses': train_losses,
                               'test_accuracies': test_accuracies, 'test_losses': test_losses}

    # 绘制结果
    plot_results(results)
    # 保存结果到 CSV 文件
    save_results_to_csv(results, output_path='training_results.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train multiple models on a dataset.")
    parser.add_argument('--dataset_dir', type=str, required=True, help="数据集文件路径")
    parser.add_argument('--models', nargs='+', default=['resnet18'],
                        help="需要训练的模型列表(如resnet18 resnet50 efficientnet_b0).")
    parser.add_argument('--batch_size', type=int, default=16, help="训练的Batch size(默认32)")
    parser.add_argument('--epochs', type=int, default=5, help="训练的轮数(默认5轮)")
    args = parser.parse_args()
    print(torch.cuda.is_available())
    main(args)

"""
python train.py --dataset_dir leaf_diseasesA_split --models resnet18
python train.py --dataset_dir leaf_diseasesA_split --models resnet18 resnet50 alexnet
python train.py --dataset_dir leaf_diseasesA_split --models resnet18 alexnet --batch_size 16 --epochs 20

"""
