from src.models.unet_resnet34 import get_unet_resnet34

import torch
from torch.utils.data import DataLoader, random_split

from src.datasets.dataset import ChangeDetectionDataset
from src.models.losses import get_combined_loss


def train():

    tiles_dirs = [
        "data/tiles_par1",
        "data/tiles_par2",
        "data/tiles_par3"
    ]

    labels_dirs = [
        "data/labels_par1",
        "data/labels_par2",
        "data/labels_par3"
    ]

    dataset = ChangeDetectionDataset(tiles_dirs, labels_dirs)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_unet_resnet34(in_channels=18, out_channels=1).to(device)

    criterion = get_combined_loss(2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_loss = float("inf")

    for epoch in range(10):

        model.train()
        train_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(
            f"Epoch {epoch+1}/{10} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "models/best_unet_resnet34.pth")


if __name__ == "__main__":
    train()