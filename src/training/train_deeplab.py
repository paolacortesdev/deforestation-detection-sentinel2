import torch
import numpy as np
from torch.utils.data import DataLoader, random_split

from src.datasets.dataset import ChangeDetectionDataset
from src.models.deeplabv3plus import get_deeplabv3plus
from src.models.losses import get_combined_loss


def train():

    # =========================
    # DATASETS
    # =========================
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
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

    # =========================
    # DEVICE
    # =========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================
    # MODEL
    # =========================
    model = get_deeplabv3plus(in_channels=18, out_channels=1).to(device)

    # =========================
    # LOSS + OPTIMIZER
    # =========================
    criterion = get_combined_loss(pos_weight=2.0)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    # =========================
    # TRAINING LOOP
    # =========================
    num_epochs = 60
    best_val_loss = float("inf")

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):

        # ---------------------
        # TRAIN
        # ---------------------
        model.train()
        train_loss = 0

        for x, y in train_loader:

            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            y_pred = model(x)

            loss = criterion(y_pred, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---------------------
        # VALIDATION
        # ---------------------
        model.eval()
        val_loss = 0

        with torch.no_grad():

            for x, y in val_loader:

                x, y = x.to(device), y.to(device)

                y_pred = model(x)

                loss = criterion(y_pred, y)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        # scheduler step
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # ---------------------
        # SAVE BEST MODEL
        # ---------------------
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                "models/best_deeplabv3plus.pth"
            )

        # ---------------------
        # LOG
        # ---------------------
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"LR: {lr:.6f}"
        )

    # =========================
    # LOAD BEST MODEL
    # =========================
    model.load_state_dict(torch.load("models/best_deeplabv3plus.pth"))
    model.eval()

    print("\nBest model loaded ✔")

    import os
    import numpy as np

    os.makedirs("results", exist_ok=True)

    np.save("results/train_losses.npy", train_losses)
    np.save("results/val_losses.npy", val_losses)

    


if __name__ == "__main__":
    train()

