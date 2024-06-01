import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import ArgoverseDataset
from encoders import MultiModalRepresentation
from formers import PETModel
from losses import TotalLoss
from trainer import Trainer

def main():
    # 配置参数
    data_root = '/path/to/argoverse/data'
    train_split = 'train'
    val_split = 'val'
    cfg = {
        'some_config_key': 'some_config_value'
    }

    # 初始化数据集
    train_dataset = ArgoverseDataset(data_root, train_split, processes=[], cfg=cfg)
    val_dataset = ArgoverseDataset(data_root, val_split, processes=[], cfg=cfg)

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 初始化模型
    model = PETModel(track_input_dim=128, track_hidden_dim=256, track_output_dim=128,
                     pm_input_dim=128, pm_hidden_dim=256, pm_output_dim=128)

    # 优化器和调度器
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 初始化Trainer
    trainer = Trainer(model, train_loader, val_loader, optimizer, scheduler, num_epochs=10)

    # 训练
    trainer.train()

if __name__ == '__main__':
    main()
