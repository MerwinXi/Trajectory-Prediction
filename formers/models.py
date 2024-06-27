import torch
import torch.nn as nn
import torch.nn.functional as F
from . import bert
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
        # 确保输入 x 是三维的
        if len(x.shape) == 4:
            x = x.squeeze(1)  # 移除多余的维度

        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])


# PMFormer: Processes planning and map data with cross-attention
class PMFormer(nn.Module):
    def __init__(self,args,history_dim,plan_dim, map_dim, output_dim):
        super(PMFormer, self).__init__()
        self.TrackFormer = bert.BERT(args)
        self.planning_encoder = nn.Linear(plan_dim, output_dim)
        self.map_encoder = nn.Linear(map_dim, output_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim=output_dim, num_heads=8)
        self.decoder = nn.Linear(output_dim, output_dim)

    def forward(self, history, map_data, plan_data):
        
        tracw = self.TrackFormer(history)

        # 确保 plan_data 的形状是 [batch_size, plan_dim]
        if len(plan_data.shape) == 1:
            plan_data = plan_data.unsqueeze(1)  # 添加一个维度以匹配输入要求

        # 调整以确保 plan_data 的形状与 planning_encoder 的输入相匹配
        if plan_data.shape[1] != self.planning_encoder.in_features:
            plan_data = plan_data.expand(-1, self.planning_encoder.in_features)

        print(f'plan_data shape: {plan_data.shape}')
        print(f'map_data shape: {map_data.shape}')

        plan_encoded = self.planning_encoder(plan_data).unsqueeze(0)
        map_encoded = self.map_encoder(map_data)

        # 将 map_encoded 的形状调整为三维
        map_encoded = map_encoded.permute(1, 0, 2)  # 调整维度顺序为 [sequence_length, batch_size, embedding_dim]

        print(f'plan_encoded shape: {plan_encoded.shape}')
        print(f'map_encoded shape: {map_encoded.shape}')

        attn_output, _ = self.cross_attention(plan_encoded, map_encoded, map_encoded)
        output = self.decoder(attn_output.squeeze(0))

        print(f'output shape: {output.shape}')

        return output
# PET: Integrates outputs from TrackFormer and PMFormer
class PET(nn.Module):
    def __init__(self,args, history_dim, planning_dim, map_dim, hidden_dim, output_dim):
        super(PET, self).__init__()
        self.n_agents = 11
        self.history_encoder = torch.nn.Sequential(nn.Linear(history_dim, 512),\
                                                   nn.Linear(512, 256),nn.Linear(256, output_dim))
        self.plan_encoder = torch.nn.Sequential(nn.Linear(planning_dim, 512),\
                                                nn.Linear(512, 256),nn.Linear(256, output_dim))
        self.map_encode = torch.nn.Sequential(nn.Linear(map_dim, 512),nn.Linear(512, 256),nn.Linear(256, output_dim))
        self.group_planning_encoder = torch.nn.Sequential(nn.Linear(history_dim, 512),\
                                                          nn.Linear(512, 256),nn.Linear(256, output_dim),nn.Softmax())
        self.TrackFormer1 = bert.BERT(args)
        self.TrackFormer2 = bert.BERT(args)
        self.TrackFormer3 = bert.BERT(args)
        self.TrackFormer4 = bert.BERT(args)
        self.TrackFormer5 = bert.BERT(args)
        self.prediction_header = nn.Linear(output_dim*output_dim,self.n_agents*planning_dim)  # 修改此处，将2改为100

    def forward(self, history_data, planning_data,group_planning_feature, map_data):
        # print("history_data:",history_data.shape)
        # print("planning_data:",planning_data.shape)
        # print("map_data:",map_data.shape)
        # print("group_planning_feature:",group_planning_feature.shape)
        history_emb = self.history_encoder(history_data)
        plan_emb = self.plan_encoder(planning_data)
        group_emb = self.plan_encoder(group_planning_feature)
        map_emb = self.map_encode(map_data)
        history_track = self.TrackFormer1(history_emb)
        plan_track = self.TrackFormer2(plan_emb)
        map_track = self.TrackFormer3(map_emb)
        group_track = self.TrackFormer4(group_emb)
        # print('history_track:',history_track.shape)
        # print('plan_track:',plan_track.shape)
        # print('map_track:',map_track.shape)
        plan_track = torch.transpose(plan_track,-2,-1)
        map_plan_cross = torch.matmul( plan_track, map_track)
        # print('group_track:',group_track.shape)
        # print('history_track:',history_track.shape)
        group_track = torch.transpose(group_track,-2,-1)
        history_group_cross = torch.matmul( group_track, history_track)
        # print('map_plan_cross:',map_plan_cross.shape)
        # print('history_plan_cross:',history_plan_cross.shape)
        history_group_cross_track = self.TrackFormer5(history_group_cross)
        cross_last = torch.matmul( history_group_cross, history_group_cross_track)
        prediction = self.prediction_header(cross_last.view(cross_last.shape[0],-1))
        prediction = torch.reshape(prediction,(cross_last.shape[0],self.n_agents,-1))
        return prediction







