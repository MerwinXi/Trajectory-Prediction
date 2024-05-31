import torch.nn as nn

class HistoryTrajectoryEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(HistoryTrajectoryEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, history):
        batch_size, N, Th, input_dim = history.size()
        history = history.view(batch_size * N, Th, input_dim)
        history_encoded = self.mlp(history)
        history_encoded = history_encoded.view(batch_size, N, Th, -1)
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
        batch_size, L, O, input_dim = hd_map.size()
        hd_map = hd_map.view(batch_size * L, O, input_dim).permute(0, 2, 1)
        hd_map_encoded = self.fpn(hd_map)
        hd_map_encoded = hd_map_encoded.permute(0, 2, 1).view(batch_size, L, O, -1)
        return hd_map_encoded

class PlanningTrajectoryEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PlanningTrajectoryEncoder, self).__init__()
        self.fpn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=1)
        )

    def forward(self, planning_trajectory):
        batch_size, N, U, T, input_dim = planning_trajectory.size()
        planning_trajectory = planning_trajectory.view(batch_size * N * U, T, input_dim).permute(0, 2, 1)
        planning_trajectory_encoded = self.fpn(planning_trajectory)
        planning_trajectory_encoded = planning_trajectory_encoded.permute(0, 2, 1).view(batch_size, N, U, T, -1)
        return planning_trajectory_encoded

class MultiModalRepresentation(nn.Module):
    def __init__(self, history_input_dim, history_hidden_dim, history_output_dim,
                 map_input_dim, map_hidden_dim, map_output_dim,
                 planning_input_dim, planning_hidden_dim, planning_output_dim):
        super(MultiModalRepresentation, self).__init__()
        self.history_encoder = HistoryTrajectoryEncoder(history_input_dim, history_hidden_dim, history_output_dim)
        self.map_encoder = HDMapEncoder(map_input_dim, map_hidden_dim, map_output_dim)
        self.planning_encoder = PlanningTrajectoryEncoder(planning_input_dim, planning_hidden_dim, planning_output_dim)

    def forward(self, history, hd_map, planning_trajectory):
        history_encoded = self.history_encoder(history)
        hd_map_encoded = self.map_encoder(hd_map)
        planning_trajectory_encoded = self.planning_encoder(planning_trajectory)
        return history_encoded, hd_map_encoded, planning_trajectory_encoded
