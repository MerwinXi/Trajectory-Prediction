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

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, num_epochs=10):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        for data in self.train_loader:
            history, future, map_features, planning_features, target = [d.to(self.device) for d in data]
            self.optimizer.zero_grad()
            output = self.model(history, map_features, planning_features)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
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
                history, future, map_features, planning_features, target = [d.to(self.device) for d in data]
                output = self.model(history, map_features, planning_features)
                loss = self.loss_fn(output, target)
                total_loss += loss.item()

                # 计算各类评估指标
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

    def save_checkpoint(self, epoch, filename="checkpoint.pth"):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        torch.save(state, filename)

    def load_checkpoint(self, filename="checkpoint.pth"):
        state = torch.load(filename)
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.scheduler.load_state_dict(state['scheduler_state_dict'])
        start_epoch = state['epoch'] + 1
        return start_epoch

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
                self.save_checkpoint(epoch)







