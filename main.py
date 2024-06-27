import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import TrajectoryDataset
from formers.models import PET
from trainer import Trainer
import config

def main():
    # 配置参数
    args = config.config()
    num_samples = 1000
    seq_length = 10
    feature_dim = 40   # 原来的 input_dim 对应 feature_dim
    drivable_area_dim = 8268*2
    map_dim = 500*2
    planning_dim = 60

    train_dataset = TrajectoryDataset(args,num_samples=num_samples, seq_length=seq_length, feature_dim=feature_dim, drivable_area_dim=drivable_area_dim)
    val_dataset = TrajectoryDataset(args,num_samples=num_samples, seq_length=seq_length, feature_dim=feature_dim, drivable_area_dim=drivable_area_dim)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = PET(args,history_dim=feature_dim, planning_dim=planning_dim, map_dim=map_dim, hidden_dim=256, output_dim=feature_dim)

    optimizer = optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    trainer = Trainer(model, train_loader, val_loader, optimizer, scheduler, num_epochs=10)
    trainer.train()

if __name__ == '__main__':
    main()
