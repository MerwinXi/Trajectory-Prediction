import torch
import torch.nn as nn

class SimpleEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        encoded = self.fc(hidden[-1])
        return encoded

class CrossModalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CrossModalEncoder, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=8, dim_feedforward=hidden_dim),
            num_layers=6
        )
        self.cross_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=8)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, src, tgt):
        memory = self.encoder(src)
        cross_attention_output, _ = self.cross_attention(tgt, memory, memory)
        return self.fc(cross_attention_output)

class SimpleDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleDecoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        decoded = self.fc(hidden[-1])
        return decoded

class PreFusionDModel(nn.Module):
    def __init__(self, track_input_dim, track_hidden_dim, track_output_dim,
                 map_input_dim, map_hidden_dim, map_output_dim,
                 planning_input_dim, planning_hidden_dim, planning_output_dim):
        super(PreFusionDModel, self).__init__()
        # Track encoding and decoding
        self.track_encoder = SimpleEncoder(track_input_dim, track_hidden_dim, track_output_dim)
        self.track_decoder = SimpleDecoder(track_output_dim, track_hidden_dim, track_output_dim)

        # Map and Planning encoding
        self.map_encoder = SimpleEncoder(map_input_dim, map_hidden_dim, map_output_dim)
        self.planning_encoder = SimpleEncoder(planning_input_dim, planning_hidden_dim, planning_output_dim)

        # Cross-modal encoding
        self.cross_modal_encoder = CrossModalEncoder(planning_input_dim, planning_hidden_dim, planning_output_dim)

        # Prediction header
        self.prediction_header = nn.Linear(track_output_dim, 2)  # For final prediction (x, y) coordinates

    def forward(self, track_embedding, map_embedding, planning_embedding):
        # Track encoding and decoding
        track_encoded = self.track_encoder(track_embedding)
        track_decoded = self.track_decoder(track_encoded)

        # Map and Planning encoding
        map_encoded = self.map_encoder(map_embedding)
        planning_encoded = self.planning_encoder(planning_embedding)

        # Cross-modal encoding
        cross_modal_encoded = self.cross_modal_encoder(planning_encoded, map_encoded)

        # Final prediction
        final_output = self.prediction_header(cross_modal_encoded)
        return final_output

