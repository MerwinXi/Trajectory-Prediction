import torch
import torch.nn as nn

class TrackFormer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=6):
        super(TrackFormer, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=8, dim_feedforward=hidden_dim),
            num_layers=num_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=input_dim, nhead=8, dim_feedforward=hidden_dim),
            num_layers=num_layers
        )
        self.fc = nn.Linear(input_dim, output_dim)
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=8)

    def forward(self, src, tgt):
        memory = self.encoder(src)
        attention_output, _ = self.attention(memory, memory, memory)
        output = self.decoder(tgt, attention_output)
        return self.fc(output)

class PMFormer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=6):
        super(PMFormer, self).__init__()
        self.map_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=8, dim_feedforward=hidden_dim),
            num_layers=num_layers
        )
        self.planning_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=8, dim_feedforward=hidden_dim),
            num_layers=num_layers
        )
        self.cross_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=8)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, map_features, planning_features):
        map_memory = self.map_encoder(map_features)
        planning_memory = self.planning_encoder(planning_features)
        cross_attention_output, _ = self.cross_attention(planning_memory, map_memory, map_memory)
        return self.fc(cross_attention_output)

class PETModel(nn.Module):
    def __init__(self, track_input_dim, track_hidden_dim, track_output_dim,
                 pm_input_dim, pm_hidden_dim, pm_output_dim):
        super(PETModel, self).__init__()
        self.trackformer = TrackFormer(track_input_dim, track_hidden_dim, track_output_dim)
        self.pmformer = PMFormer(pm_input_dim, pm_hidden_dim, pm_output_dim)
        self.prediction_header = nn.Linear(track_output_dim, pm_output_dim)
        self.uncertainty_module = nn.Linear(pm_output_dim, 2)  # GMM的μ和σ

    def forward(self, history, future, map_features, planning_features):
        track_output = self.trackformer(history, future)
        pm_output = self.pmformer(map_features, planning_features)
        final_output = self.prediction_header(pm_output)
        mu, sigma = self.uncertainty_module(final_output).chunk(2, dim=-1)
        return final_output, mu, sigma
