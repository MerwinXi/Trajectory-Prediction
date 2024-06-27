import torch
import numpy as np
from metrics import (
    min_ade,
    min_fde,
    miss_rate,
    brier_min_fde,
    brier_min_ade,
    drivable_area_compliance,
)
from losses.TotalLoss import CustomLossModule  # 导入损失函数模块

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, loss_fn=None, num_epochs=10):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn if loss_fn is not None else CustomLossModule()
        self.num_epochs = num_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        print('self.train_loader:',len(self.train_loader))
        for data in self.train_loader:
            history, future, map_features, planning_features, group_planning_feature, target, drivable_mask = [d.to(self.device).float() for d in data]
            # print(f'history shape: {history.shape}')
            # print(f'future shape: {future.shape}')
            # print(f'map_features shape: {map_features.shape}')
            # print(f'planning_features shape: {planning_features.shape}')
            # print(f'group_planning_feature shape: {group_planning_feature.shape}')
            # print(f'target shape: {target.shape}')
            history = history.view(history.shape[0],11,-1)
            future = future.view(history.shape[0],11,-1)
            map_features = map_features.view(history.shape[0],-1)
            planning_features = planning_features.view(history.shape[0],-1)
            group_planning_feature = group_planning_feature.view(history.shape[0], 11, -1)
            # target = torch.tensor(target)
            # drivable_mask = torch.tensor(drivable_mask)

            self.optimizer.zero_grad()
            output = self.model(history, planning_features,group_planning_feature, map_features)
            loss = self.loss_fn(output, future, drivable_mask,future)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            # print('total_loss:',total_loss)
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def validate(self):
        self.model.eval()
        total_loss = 0
        metrics = {
            "minADE": [],
            "minFDE": [],
            "MissRate": [],
            "Brier-minFDE": [],
            "Brier-minADE": [],
            "DAC": []
        }
        with torch.no_grad():
            for data in self.val_loader:
                history, future, map_features, planning_features, target, drivable_mask, labels = [d.to(self.device).float() for d in data]
                print(f'validate history shape: {history.shape}')
                print(f'validate future shape: {future.shape}')
                print(f'validate map_features shape: {map_features.shape}')
                print(f'validate planning_features shape: {planning_features.shape}')
                print(f'validate target shape: {target.shape}')

                output = self.model(history, planning_features, map_features)
                loss = self.loss_fn(output, target, drivable_mask, labels)
                total_loss += loss.item()

                predictions = output.cpu().numpy()
                ground_truth = target.cpu().numpy()
                probabilities = np.ones(predictions.shape[0])

                metrics["minADE"].append(min_ade(predictions, ground_truth))
                metrics["minFDE"].append(min_fde(predictions, ground_truth))
                metrics["MissRate"].append(miss_rate(predictions, ground_truth))
                metrics["Brier-minFDE"].append(brier_min_fde(predictions, ground_truth, probabilities))
                metrics["Brier-minADE"].append(brier_min_ade(predictions, ground_truth, probabilities))
                metrics["DAC"].append(drivable_area_compliance(predictions, map_features.cpu().numpy()))

        avg_loss = total_loss / len(self.val_loader)
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
        return avg_loss, avg_metrics

    def save_checkpoint(self, epoch, best_val_loss, filename="checkpoint.pth"):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': best_val_loss
        }
        torch.save(state, filename)

    def load_checkpoint(self, filename="checkpoint.pth"):
        state = torch.load(filename)
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.scheduler.load_state_dict(state['scheduler_state_dict'])
        return state['epoch'], state.get('best_val_loss', float('inf'))

    def train(self):
        best_val_loss = float('inf')
        for epoch in range(self.num_epochs):
            train_loss = self.train_one_epoch()
            val_loss, val_metrics = self.validate()
            self.scheduler.step()

            print(f"Epoch {epoch + 1}/{self.num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
            print(f"Validation Metrics: {val_metrics}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, best_val_loss)
