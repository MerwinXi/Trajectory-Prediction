import torch
import torch.nn as nn

class HistoryTrajectoryEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(HistoryTrajectoryEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, history):
        _, (hidden, _) = self.lstm(history)
        history_encoded = self.fc(hidden[-1])
        return history_encoded

class HDMapEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(HDMapEncoder, self).__init__()
        self.fpn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=1)
        )

    def forward(self, hd_map):
        hd_map_encoded = self.fpn(hd_map)
        return hd_map_encoded

class PlanningTrajectoryEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PlanningTrajectoryEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, planning_trajectory):
        _, (hidden, _) = self.lstm(planning_trajectory)
        planning_trajectory_encoded = self.fc(hidden[-1])
        return planning_trajectory_encoded

class MultiModalRepresentation(nn.Module):
    def __init__(self, history_input_dim, history_hidden_dim, history_output_dim,
                 map_input_dim, map_hidden_dim, map_output_dim,
                 planning_input_dim, planning_hidden_dim, planning_output_dim):
        super(MultiModalRepresentation, self).__init__()
        self.history_encoder = HistoryTrajectoryEncoder(history_input_dim, history_hidden_dim, history_output_dim)
        self.map_encoder = HDMapEncoder(map_input_dim, map_hidden_dim, map_output_dim)
        self.planning_encoder = PlanningTrajectoryEncoder(planning_input_dim, planning_hidden_dim, planning_output_dim)
        self.fusion_layer = nn.Linear(history_output_dim + map_output_dim + planning_output_dim, 512)

    def forward(self, history, hd_map, planning_trajectory):
        history_encoded = self.history_encoder(history)
        hd_map_encoded = self.map_encoder(hd_map)
        planning_trajectory_encoded = self.planning_encoder(planning_trajectory)
        
        combined_features = torch.cat((history_encoded, hd_map_encoded, planning_trajectory_encoded), dim=-1)
        fused_features = self.fusion_layer(combined_features)
        return fused_features



