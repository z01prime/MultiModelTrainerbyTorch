# coding: utf-8
import time
import os
import numpy as np
import cv2  # opencv-python
from PIL import Image, ImageFont, ImageDraw
import torch
import torch.nn.functional as F
from torchvision import transforms
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class RealTimeDetector:
    def __init__(self, model_path, idx_to_labels_path, font_path=None, device=None):
        """
        初始化实时检测器
        :param model_path: 模型文件路径
        :param idx_to_labels_path: 类别索引到标签的映射文件路径
        :param font_path: 字体文件路径
        :param device: 设备类型（cuda 或 cpu）
        """
        # 设置设备
        self.device = device if device else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Using device:', self.device)

        # 加载模型
        self.model = torch.load(model_path, map_location=self.device)
        self.model = self.model.eval().to(self.device)

        # 加载类别映射
        self.idx_to_labels = np.load(idx_to_labels_path, allow_pickle=True).item()

        # 加载字体
        if font_path and os.path.exists(font_path):
            self.font = ImageFont.truetype(font_path, 16)
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

    def process_frame(self, img):
        """
        处理单帧图像，进行预测和结果绘制
        :param img: 输入帧（BGR 格式）
        :return: 处理后的帧（BGR 格式）
        """
        start_time = time.time()

        # BGR 转 RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # 图像预处理
        input_img = self.test_transform(img_pil).unsqueeze(0).to(self.device)

        # 模型预测
        pred_logits = self.model(input_img)
        pred_softmax = F.softmax(pred_logits, dim=1)

        # 获取 Top-N 预测结果
        top_n = torch.topk(pred_softmax, min(5,len(self.idx_to_labels)))
        pred_ids = top_n[1].cpu().detach().numpy().squeeze()
        confs = top_n[0].cpu().detach().numpy().squeeze()

        # 绘制预测结果
        draw = ImageDraw.Draw(img_pil)
        for i in range(len(confs)):
            pred_class = self.idx_to_labels[pred_ids[i]]
            text = '{:<15} {:>.3f}'.format(pred_class, confs[i])
            draw.text((50, 100 + 50 * i), text, font=self.font, fill=(255, 0, 0, 1))

        # PIL 转 BGR
        img = np.array(img_pil)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 计算 FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        img = cv2.putText(img, f'FPS {int(fps)}', (50, 80),
                          cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 4, cv2.LINE_AA)
        return img

    def run(self, camera_id=0):
        """
        启动摄像头进行实时检测
        :param camera_id: 摄像头设备 ID（默认 0 为系统默认摄像头）
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("Error: Unable to access the camera.")
            return

        print("Press 'q' or 'ESC' to exit.")
        while True:
            success, frame = cap.read()
            if not success:
                print("Error: Unable to read the frame.")
                break

            # 处理帧
            frame = self.process_frame(frame)

            # 显示处理后的帧
            cv2.imshow('Real-Time Detection', frame)

            # 按 'q' 或 'ESC' 退出
            if cv2.waitKey(1) in [ord('q'), 27]:
                break

        cap.release()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time Object Detection using Camera")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model file")
    parser.add_argument('--idx_to_labels_path', type=str, required=True, help="Path to the idx_to_labels.npy file")
    parser.add_argument('--camera_id', type=int, default=0, help="ID of the camera (default is 0)")
    parser.add_argument('--font_path', type=str, default=None,
                        help="Path to the font file (default is system default font)")
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu',
                        help="Device type (default is 'cpu')")
    return parser.parse_args()


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()

    # 初始化实时检测器
    detector = RealTimeDetector(
        model_path=args.model_path,
        idx_to_labels_path=args.idx_to_labels_path,
        font_path=args.font_path,
        device=torch.device(args.device)
    )

    # 启动摄像头检测
    detector.run(camera_id=args.camera_id)

