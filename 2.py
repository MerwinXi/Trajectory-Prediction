import torch
import torch.nn as nn
import torch.nn.functional as F

class HistoryTrajectoryEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, (hidden, _) = self.lstm(x)
        hidden = hidden[-1]  # Taking the last layer's hidden state
        return self.fc(hidden)

class PlanningTrajectoryEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class HDMapEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# TrackFormer: Uses LSTM to process tracking data
class TrackFormer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TrackFormer, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])

# PMFormer: Processes planning and map data with cross-attention
class PMFormer(nn.Module):
    def __init__(self, plan_dim, map_dim, output_dim):
        super(PMFormer, self).__init__()
        self.planning_encoder = nn.Linear(plan_dim, output_dim)
        self.map_encoder = nn.Linear(map_dim, output_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim=output_dim, num_heads=8)
        self.decoder = nn.Linear(output_dim, output_dim)

    def forward(self, plan_data, map_data):
        plan_encoded = self.planning_encoder(plan_data).unsqueeze(0)
        map_encoded = self.map_encoder(map_data).unsqueeze(0)
        attn_output, _ = self.cross_attention(plan_encoded, map_encoded, map_encoded)
        output = self.decoder(attn_output.squeeze(0))
        return output

# PET: Integrates outputs from TrackFormer and PMFormer
class PET(nn.Module):
    def __init__(self, history_dim, planning_dim, map_dim, hidden_dim, output_dim):
        super(PET, self).__init__()
        self.history_encoder = nn.Linear(history_dim, output_dim)
        self.pmformer = PMFormer(planning_dim, map_dim, output_dim)
        self.trackformer = TrackFormer(output_dim, hidden_dim, output_dim)
        self.prediction_header = nn.Linear(output_dim * 2, 2)

    def forward(self, history_data, planning_data, map_data):
        history_emb = self.history_encoder(history_data)
        pm_output = self.pmformer(planning_data, map_data)
        track_output = self.trackformer(history_emb.unsqueeze(1))
        combined_output = torch.cat((pm_output, track_output), dim=1)
        prediction = self.prediction_header(combined_output)
        return prediction




