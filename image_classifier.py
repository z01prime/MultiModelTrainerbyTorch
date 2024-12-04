# coding: utf-8
import warnings
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageFont, ImageDraw

# torchvision 和 matplotlib 之间可能有依赖加载冲突。mat要放在前面

class ImageClassifier:
    def __init__(self, model_path, idx_to_labels_path, font_path='SimHei.ttf', device=None):
        """
        初始化图片分类器
        :param model_path: 模型文件路径
        :param idx_to_labels_path: 类别索引到标签的映射文件路径
        :param font_path: 字体文件路径
        :param device: 设备类型（cuda 或 cpu）
        """
        warnings.filterwarnings("ignore")
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        # 设置设备
        self.device = device if device else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Using device:', self.device)

        # 加载模型
        self.model = torch.load(model_path, map_location=self.device).eval().to(self.device)

        # 加载类别映射
        self.idx_to_labels = np.load(idx_to_labels_path, allow_pickle=True).item()

        # 加载字体
        if os.path.exists(font_path):
            self.font = ImageFont.truetype(font_path, 32)
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
        else:
            print("Font not found. Using default font.")
            self.font = ImageFont.load_default()

        # 图像预处理
        self.test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4635, 0.4889, 0.4080], std=[0.1755, 0.1484, 0.1919])
        ])

    def predict(self, img_path, save_results=True):
        """
        对单张图片进行分类预测
        :param img_path: 图片路径
        :param save_results: 是否保存预测结果和图表
        """
        print(f"Processing image: {img_path}")

        # 加载和预处理图像
        img_pil = Image.open(img_path)
        input_img = self.test_transform(img_pil).unsqueeze(0).to(self.device)

        # 执行模型预测
        pred_logits = self.model(input_img)
        pred_softmax = F.softmax(pred_logits, dim=1).cpu().detach().numpy()[0]

        # 获取预测结果
        pred_confidences = pred_softmax * 100
        pred_ids = np.argsort(-pred_confidences)
        results = [(self.idx_to_labels[i], pred_confidences[i]) for i in pred_ids]

        # 绘制预测结果
        self._draw_results(img_pil, results, img_path, save_results)

        # 保存结果为表格
        if save_results:
            self._save_predictions(results, img_path)

        return results

    def _draw_results(self, img_pil, results, img_path, save_results):
        """
        绘制预测结果到图像和柱状图
        :param img_pil: 原始 PIL 图像
        :param results: 预测结果 (类别名称, 置信度)
        :param img_path: 原始图片路径
        :param save_results: 是否保存结果图
        """
        draw = ImageDraw.Draw(img_pil)
        for i, (class_name, confidence) in enumerate(results):
            text = f"{class_name:<15} {confidence:.2f}%"
            draw.text((50, 100 + 50 * i), text, font=self.font, fill=(255, 0, 0, 1))

        # 绘制柱状图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
        ax1.imshow(img_pil)
        ax1.axis('off')

        classes = [item[0] for item in results]
        confidences = [item[1] for item in results]
        bars = ax2.bar(classes, confidences, alpha=0.7, color='yellow', edgecolor='red', lw=2,width = 0.4)
        ax2.bar_label(bars, fmt="%.2f%%", fontsize=10)
        ax2.set_xlabel("类别", fontsize=16)
        ax2.set_ylabel("置信度 (%)", fontsize=16)
        ax2.set_title(f"{img_path} 预测结果", fontsize=20)
        ax2.tick_params(axis='x', rotation=90, labelsize=8)

        plt.tight_layout()
        if save_results:
            output_path = "output/预测图+柱状图.jpg"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.savefig(output_path)
            print(f"Saved results figure to {output_path}")
        plt.show()

    def _save_predictions(self, results, img_path):
        """
        保存预测结果到 CSV 文件
        :param results: 预测结果 (类别名称, 置信度)
        :param img_path: 原始图片路径
        """
        pred_df = pd.DataFrame(results, columns=["Class", "Confidence (%)"])
        output_path = "output/predictions.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pred_df.to_csv(output_path, index=False)
        print(f"Saved predictions to {output_path}")


def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='Image Classification CLI')
    parser.add_argument('--image_path', type=str, help='Path to the image to be classified')
    parser.add_argument('--model', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--labels', type=str, default='idx_to_labels.npy', help='Path to the labels mapping')
    parser.add_argument('--font', type=str, default='SimHei.ttf', help='Path to the font file')
    parser.add_argument('--no-save', action='store_true', help='Disable saving results')

    # 解析参数
    args = parser.parse_args()

    # 初始化分类器
    classifier = ImageClassifier(
        model_path=args.model,
        idx_to_labels_path=args.labels,
        font_path=args.font
    )

    # 对图片进行分类
    results = classifier.predict(img_path=args.image_path, save_results=not args.no_save)
    print("预测结果:", results)


if __name__ == "__main__":
    main()