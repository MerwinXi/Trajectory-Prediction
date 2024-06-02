import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TrajectoryDataset(Dataset):
    def __init__(self, num_samples=1000, seq_length=10, feature_dim=2, drivable_area_dim=100):
        # Simulate historical trajectories
        self.history_trajectories = np.random.randn(num_samples, seq_length, feature_dim)
        # Simulate future real trajectories
        self.ground_truth = self.history_trajectories + np.random.randn(num_samples, seq_length, feature_dim) * 0.1
        # Simulate predicted trajectories
        self.predictions = self.history_trajectories + np.random.randn(num_samples, seq_length, feature_dim) * 0.1
        # Simulate prediction probabilities
        self.probabilities = np.random.rand(num_samples)
        # Simulate drivable area masks
        self.drivable_area_mask = np.random.randint(0, 2, (num_samples, drivable_area_dim))

    def __len__(self):
        return len(self.history_trajectories)

    def __getitem__(self, idx):
        # Convert all numpy data to tensors
        history = torch.from_numpy(self.historytrajectories[idx]).float()
        ground_truth = torch.from_numpy(self.ground_truth[idx]).float()
        predictions = torch.from_numpy(self.predictions[idx]).float()
        probabilities = torch.from_numpy(self.probabilities[idx]).float()
        drivable_mask = torch.from_numpy(self.drivable_area_mask[idx]).float()
        
        return history, ground_truth, predictions, probabilities, drivable_mask

