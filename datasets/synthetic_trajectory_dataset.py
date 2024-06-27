from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import sys
import os
sys.path.append(os.path.abspath('../'))
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
import warnings
warnings.catch_warnings()
warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.simplefilter('ignore', np.RankWarning)
class TrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, args,num_samples, seq_length, feature_dim, drivable_area_dim):
        # 初始化数据集
        self.map = args.map_path
        self.data = args.train_path
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.feature_dim = feature_dim
        self.drivable_area_dim = drivable_area_dim
        self.af = ArgoverseForecastingLoader(self.data)
        # 初始化 Argoverse 地图接口
        self.avm = ArgoverseMap(self.map)

    def _generate_sample(self):
        history = torch.randn(self.seq_length, self.feature_dim)
        future = torch.randn(self.seq_length, self.feature_dim)
        map_features = torch.randn(self.seq_length, self.feature_dim)
        planning_features = torch.randn(self.feature_dim)
        target = torch.randn(100)
        drivable_mask = torch.randint(0, 2, (self.seq_length,))  # 修改此处，将 feature_dim 改为 1
        labels = torch.randint(0, 2, (100,))
        return history, future, map_features, planning_features, target, drivable_mask, labels

    def __len__(self):
        return len(self.af)

    def __getitem__(self, idx):
        seq_path  = self.af.seq_list[idx]
        df = self.af.get(seq_path ).seq_df
        object_ids = df["TRACK_ID"].unique()
        max_x = 0
        max_y = 0
        # 获取时间步数（假设每个对象都有相同的时间步数）
        # 这里假设我们取每个对象的所有时间步数，如果时间步数不一致，可以进一步处理
        time_steps = df[df["TRACK_ID"] == object_ids[0]].shape[0]
        # 初始化轨迹数组，形状为 [agent数量, 时间步数, 2]
        trajectories = np.zeros((11, time_steps, 2))
        # print(trajectories.shape)
        # 填充轨迹数据
        for i, object_id in enumerate(object_ids):
            if i<11:
                object_df = df[(df["TRACK_ID"] == object_id) & (df["OBJECT_TYPE"] != "AGENT")]
                trajectories[i, :len(object_df), 0] = object_df["X"].values
                trajectories[i, :len(object_df), 1] = object_df["Y"].values

        # 获取 agent 所在城市
        city_name = df.iloc[0]["CITY_NAME"]
        agent_df = df[df["OBJECT_TYPE"] == "AGENT"]
        point_20 = agent_df.iloc[19][["X", "Y"]].values

        # 获取所有车道节点
        lane_centerlines = self.avm.city_lane_centerlines_dict[city_name]
        # print(point_20)
        map_features = self.get_lane_nodes_within_radius(point_20,100,lane_centerlines)
               
        agent_obs_traj = self.af.get(seq_path).agent_traj
        trajectories[-1, :len(agent_obs_traj), :] = agent_obs_traj
        target = trajectories[-1, -1, :]
        history = trajectories[:,:20,:]
        future = trajectories[:,-30:,:]
        planning_features = self.avm.get_candidate_centerlines_for_traj(agent_obs_traj, self.af[idx].city, viz=False) 
        if type(planning_features) is list:
            planning_features = planning_features[0]
            if planning_features.shape[0]>30:
                planning_features = planning_features[-30:,:]
            else:
                tmp = np.zeros((30,2))
                tmp[:planning_features.shape[0],:] = planning_features
                planning_features = tmp
            mx = np.max(planning_features[:,0])
            my = np.max(planning_features[:,1])
            if mx > max_x:max_x = mx
            if my > max_y:max_y = my
        drivable_area = self.avm.find_local_driveable_areas([agent_obs_traj[20,0]-50, agent_obs_traj[20,0]+50, agent_obs_traj[20,1]-50, agent_obs_traj[20,1]+50], 'PIT')
        group_planning_feature = []
        for i in range(11):
            try:
                planning = self.avm.get_candidate_centerlines_for_traj(trajectories[i,:,:], self.af[idx].city, viz=False) 
            except AssertionError:
                planning = np.zeros((30,2))
            if type(planning) is list:
                planning = planning[0]
                if planning.shape[0]>30:
                    planning = planning[-30:,:2]
                else :
                    tmp = np.zeros((30,2))
                    tmp[:planning.shape[0],:] = planning
                    planning = tmp
            mx = np.max(planning[:,0])
            my = np.max(planning[:,1])
            if mx > max_x:max_x = mx
            if my > max_y:max_y = my                    
            # print('plan:',planning.shape)
            group_planning_feature.append(planning)
        group_planning_feature = np.stack(group_planning_feature)
        if len(drivable_area)>0:
            # print(drivable_area[0].shape)
            drivable_area = drivable_area[0][:,0:2]
        else:
            drivable_area = np.zeros((8268,2))
        mx = np.max(drivable_area[:,0])
        my = np.max(drivable_area[:,1])
        if mx > max_x:max_x = mx
        if my > max_y:max_y = my
        if map_features.shape[0]>500:
            # print(drivable_area[0].shape)
            map_features = drivable_area[:500,:2]
        else:
            map_features1 = np.zeros((500,2))
            map_features1[:map_features.shape[0],:] = map_features[:,0:2]
            map_features = map_features1
        mx = np.max(map_features[:,0])
        my = np.max(map_features[:,1])
        if mx > max_x:max_x = mx
        if my > max_y:max_y = my
        history[:,:,0] = history[:,:,0] / max_x   
        history[:,:,1] = history[:,:,1] / max_y  
        future[:,:,0] = future[:,:,0] / max_x   
        future[:,:,1] = future[:,:,1] / max_y   
        map_features[:,0] = map_features[:,0] / max_x   
        map_features[:,1] = map_features[:,1] / max_y  
        planning_features[:,0] = planning_features[:,0] / max_x   
        planning_features[:,1] = planning_features[:,1] / max_y   
        group_planning_feature[:,:,0] = group_planning_feature[:,:,0] / max_x   
        group_planning_feature[:,:,1] = group_planning_feature[:,:,1] / max_y  
        drivable_area[:,0] = drivable_area[:,0] / max_x   
        drivable_area[:,1] = drivable_area[:,1] / max_y            
        return history, future, map_features, \
            planning_features, group_planning_feature, target ,drivable_area
        # 将 possible_routes 转换为 x, y 坐标
    def routes_to_coordinates(self,possible_routes, lane_centerlines):
        routes_coordinates = []
        for route in possible_routes:
            route_coordinates = []
            for lane_id in route:
                lane = lane_centerlines[lane_id]
                route_coordinates.extend(lane.centerline.tolist())
            routes_coordinates.append(np.array(route_coordinates))
        return routes_coordinates
    # 获取半径 R 内的车道节点坐标
    def get_lane_nodes_within_radius(self,agent_pos, R, lane_centerlines):
        # nearby_lane_nodes = []
        # Map_representation =  np.zeros((len(object_ids), time_steps, 2))
        Map_representation = []
        agent_x, agent_y = agent_pos
        for lane_id, lane in lane_centerlines.items():
            first_ = []
            for x, y in lane.centerline:            
                distance = np.sqrt((x - agent_x) ** 2 + (y - agent_y) ** 2)
                # print(distance)
                if distance <= R:
                    # nearby_lane_nodes.append((x, y))
                    first_.append([x, y])
                
            if len(first_)>0:Map_representation.append(first_)
        return np.vstack(Map_representation)
    # 定义一个函数来计算Frenet坐标系
    def get_frenet_coordinates(self,x, y, kdtree, lane_centerlines,all_lane_points):
        _, idx = kdtree.query([x, y])
        nearest_point = all_lane_points[idx]
        for lane_id, lane in lane_centerlines.items():
            if any(np.all(nearest_point == lane.centerline, axis=1)):
                break
        s = np.linalg.norm(nearest_point - lane.centerline[0])
        d = np.linalg.norm([x, y] - nearest_point)
        return s, d, lane_id
    def find_possible_routes(self,start_lane, end_lane, lane_centerlines,avm , max_depth=5):
        routes = []
        # visited = set()

        def dfs(current_lane, path, depth):
            if depth > max_depth:
                return
            # if current_lane in visited:
                # return
            # visited.add(current_lane)

            if current_lane == end_lane:
                if self.check_conditions(path + [current_lane], lane_centerlines,avm):
                    routes.append(path + [current_lane])
                return
            
            if hasattr(lane_centerlines[current_lane], 'successors'):
                if lane_centerlines[current_lane].successors is None:
                    return
                for successor in lane_centerlines[current_lane].successors:
                    if successor is None:
                        continue
                    dfs(successor, path + [current_lane], depth + 1)
            # visited.remove(current_lane)

        dfs(start_lane, [], 0)
        return routes
    # 条件检查函数
    def check_conditions(self,path, lane_centerlines,avm):
        for i in range(len(path) - 1):
            current_lane = path[i]
            next_lane = path[i + 1]
            # 检查是否存在急转弯
            direction_change = np.arccos(np.dot(
                lane_centerlines[current_lane].centerline[-1] - lane_centerlines[current_lane].centerline[-2],
                lane_centerlines[next_lane].centerline[1] - lane_centerlines[next_lane].centerline[0]
            ))
            if direction_change > np.pi / 4:  # 急转弯的阈值，可以调整
                return False
            
            # 检查是否驶出可驾驶区域
            if not avm.is_in_av_drivable_area(lane_centerlines[current_lane].centerline[-1], city_name) or \
            not avm.is_in_av_drivable_area(lane_centerlines[next_lane].centerline[0], city_name):
                return False
            
            # 检查是否有高速度换道
            if np.linalg.norm(lane_centerlines[current_lane].centerline[-1] - lane_centerlines[next_lane].centerline[0]) > 5.0:  # 换道距离阈值，可以调整
                return False
            
            # 检查高加加速度（jerk）
            if i >= 2:
                prev_lane = path[i - 1]
                jerk = np.linalg.norm(
                    lane_centerlines[next_lane].centerline[0] - 2 * lane_centerlines[current_lane].centerline[-1] + lane_centerlines[prev_lane].centerline[-1]
                )
                if jerk > 10.0:  # 加加速度阈值，可以调整
                    return False
                
        return True




