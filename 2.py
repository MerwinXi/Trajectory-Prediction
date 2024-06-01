import torch
import torch.nn as nn

class SelfAttentionEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=6):
        super(SelfAttentionEncoder, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=8, dim_feedforward=hidden_dim),
            num_layers=num_layers
        )

    def forward(self, x):
        return self.encoder(x)

class CrossAttentionEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=6):
        super(CrossAttentionEncoder, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=8, dim_feedforward=hidden_dim),
            num_layers=num_layers
        )
        self.cross_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=8)

    def forward(self, src, tgt):
        memory = self.encoder(src)
        cross_attention_output, _ = self.cross_attention(tgt, memory, memory)
        return cross_attention_output

class TrackFormer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TrackFormer, self).__init__()
        self.self_attention1 = SelfAttentionEncoder(input_dim, hidden_dim)
        self.self_attention2 = SelfAttentionEncoder(input_dim, hidden_dim)
        self.cross_attention = CrossAttentionEncoder(input_dim, hidden_dim)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, track_embedding):
        x = self.self_attention1(track_embedding)
        x = self.self_attention2(x)
        x = self.cross_attention(x, x)
        return self.fc(x)

class PMFormer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PMFormer, self).__init__()
        self.self_attention_planning = SelfAttentionEncoder(input_dim, hidden_dim)
        self.self_attention_map = SelfAttentionEncoder(input_dim, hidden_dim)
        self.cross_attention = CrossAttentionEncoder(input_dim, hidden_dim)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, planning_embedding, map_embedding):
        planning = self.self_attention_planning(planning_embedding)
        map_ = self.self_attention_map(map_embedding)
        cross_attention_output = self.cross_attention(map_, planning)
        return self.fc(cross_attention_output)

class PETModel(nn.Module):
    def __init__(self, track_input_dim, track_hidden_dim, track_output_dim,
                 pm_input_dim, pm_hidden_dim, pm_output_dim):
        super(PETModel, self).__init__()
        self.trackformer = TrackFormer(track_input_dim, track_hidden_dim, track_output_dim)
        self.pmformer = PMFormer(pm_input_dim, pm_hidden_dim, pm_output_dim)
        self.prediction_header = nn.Linear(pm_output_dim, 2)  # For final prediction (x, y) coordinates

    def forward(self, track_embedding, map_embedding, planning_embedding):
        track_output = self.trackformer(track_embedding)
        pm_output = self.pmformer(planning_embedding, map_embedding)
        final_output = self.prediction_header(pm_output)
        return final_output




