from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

class TrajectoryDataset(Dataset):
    def __init__(self, num_samples=1000, seq_length=10, feature_dim=2, drivable_area_dim=100):
        # 模拟历史轨迹
        self.history_trajectories = np.random.randn(num_samples, seq_length, feature_dim)
        # 模拟未来真实轨迹
        self.ground_truth = self.history_trajectories + np.random.randn(num_samples, seq_length, feature_dim) * 0.1
        # 模拟预测轨迹
        self.predictions = self.history_trajectories + np.random.randn(num_samples, seq_length, feature_dim) * 0.1
        # 模拟预测概率
        self.probabilities = np.random.rand(num_samples)
        # 模拟可行驶区域掩码
        self.drivable_area_mask = np.random.randint(0, 2, (num_samples, drivable_area_dim))

    def __len__(self):
        return len(self.history_trajectories)

    def __getitem__(self, idx):
        return (self.history_trajectories[idx],
                self.ground_truth[idx],
                self.predictions[idx],
                self.probabilities[idx],
                self.drivable_area_mask[idx])

# 示例使用数据加载
dataset = TrajectoryDataset()
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)


