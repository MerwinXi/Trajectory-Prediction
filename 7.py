import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import ArgoverseDataset
from encoders import MultiModalRepresentation
from formers import PETModel
from trainer import Trainer

def main():
    data_size = 1000
    input_dim = 128

    train_dataset = ArgoverseDataset(data_size=data_size, input_dim=input_dim)
    val_dataset = ArgoverseDataset(data_size=data_size, input_dim=input_dim)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = PETModel(track_input_dim=input_dim, track_hidden_dim=256, track_output_dim=input_dim,
                     pm_input_dim=input_dim, pm_hidden_dim=256, pm_output_dim=input_dim)

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    trainer = Trainer(model, train_loader, val_loader, optimizer, scheduler, num_epochs=10)

    trainer.train()

if __name__ == '__main__':
    main()
