# coding: utf-8
import os
import shutil
import random
import pandas as pd
import argparse  # 导入 argparse 库


class DatasetSplitter:
    def __init__(self, dataset_path, test_frac=0.2, random_seed=123):
        """
        初始化数据集划分器
        :param dataset_path: 数据集路径
        :param test_frac: 测试集比例
        :param random_seed: 随机种子，确保可复现性
        """
        self.dataset_path = dataset_path
        self.test_frac = test_frac
        self.random_seed = random_seed
        self.classes = os.listdir(dataset_path)
        self.stats = []

        # 设置随机种子
        random.seed(self.random_seed)

    def split_dataset(self):
        """
        划分数据集为训练集和验证集
        """
        print(f"Splitting dataset: {self.dataset_path}")
        print(f"Classes found: {self.classes}")

        # 创建 train 和 val 文件夹
        self._create_split_dirs()

        # 划分数据集并移动文件
        print(f"{'类别':^18} {'训练集数据个数':^18} {'测试集数据个数':^18}")
        for class_name in self.classes:
            self._split_class_data(class_name)

        # 重命名数据集文件夹
        self._rename_dataset()

        # 保存统计结果
        self._save_stats()

    def _create_split_dirs(self):
        """
        在数据集目录中创建 train 和 val 文件夹
        """
        os.makedirs(os.path.join(self.dataset_path, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.dataset_path, 'val'), exist_ok=True)

        for class_name in self.classes:
            os.makedirs(os.path.join(self.dataset_path, 'train', class_name), exist_ok=True)
            os.makedirs(os.path.join(self.dataset_path, 'val', class_name), exist_ok=True)

    def _split_class_data(self, class_name):
        """
        划分某类别的数据为训练集和验证集
        :param class_name: 类别名称
        """
        class_dir = os.path.join(self.dataset_path, class_name)
        images = os.listdir(class_dir)
        random.shuffle(images)

        # 计算测试集数量
        test_count = int(len(images) * self.test_frac)
        test_images = images[:test_count]
        train_images = images[test_count:]

        # 移动测试集文件
        for img in test_images:
            shutil.move(
                os.path.join(class_dir, img),
                os.path.join(self.dataset_path, 'val', class_name, img)
            )

        # 移动训练集文件
        for img in train_images:
            shutil.move(
                os.path.join(class_dir, img),
                os.path.join(self.dataset_path, 'train', class_name, img)
            )

        # 删除空的旧类别目录
        shutil.rmtree(class_dir)

        # 保存统计信息
        print(f"{class_name:^18} {len(train_images):^18} {len(test_images):^18}")
        self.stats.append({
            'class': class_name,
            'trainset': len(train_images),
            'testset': len(test_images)
        })

    def _rename_dataset(self):
        """
        重命名数据集文件夹
        """
        new_dataset_path = self.dataset_path + '_split'
        shutil.move(self.dataset_path, new_dataset_path)
        self.dataset_path = new_dataset_path

    def _save_stats(self):
        """
        保存数据集划分统计结果为 CSV 文件
        """
        stats_df = pd.DataFrame(self.stats)
        stats_df['total'] = stats_df['trainset'] + stats_df['testset']
        stats_df.to_csv('数据量统计.csv', index=False)
        print("Saved dataset split statistics to 数据量统计.csv")


if __name__ == "__main__":
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Split dataset into train and validation sets.")
    parser.add_argument('--dataset_path', type=str, required=True, help="数据集路径")
    parser.add_argument('--test_frac', type=float, default=0.2, help="测试集比例，默认0.2")
    parser.add_argument('--random_seed', type=int, default=123, help="随机种子，默认123")

    args = parser.parse_args()

    # 初始化数据集划分器
    splitter = DatasetSplitter(dataset_path=args.dataset_path, test_frac=args.test_frac, random_seed=args.random_seed)

    # 执行数据集划分
    splitter.split_dataset()

