import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from src.models.deeplabv3plus import get_deeplabv3plus


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# LOAD MODEL
# -------------------------
model = get_deeplabv3plus(in_channels=18, out_channels=1).to(device)
model.load_state_dict(torch.load("models/best_deeplabv3plus.pth"))
model.eval()


# -------------------------
# LOAD SAMPLE TILE
# -------------------------
tiles_dir = "data/tiles_par3"
labels_dir = "data/labels_par3"

tile_id = 0

pre = np.load(os.path.join(tiles_dir, f"tile_{tile_id}_pre.npy"))
post = np.load(os.path.join(tiles_dir, f"tile_{tile_id}_post.npy"))
label = np.load(os.path.join(labels_dir, f"tile_{tile_id}_label.npy"))


x = np.concatenate([pre, post], axis=0)
x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)


# -------------------------
# PREDICTION
# -------------------------
with torch.no_grad():
    pred = model(x)
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).cpu().numpy()[0,0]


# -------------------------
# RGB VISUAL
# -------------------------
def to_rgb(tile):
    rgb = np.stack([tile[2], tile[1], tile[0]], axis=-1)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-10)
    return rgb


rgb_pre = to_rgb(pre)
rgb_post = to_rgb(post)


# -------------------------
# PLOT FINAL
# -------------------------
plt.figure(figsize=(15,5))


plt.subplot(1,3,1)
plt.imshow(rgb_post)
plt.title("POST")
plt.axis("off")


plt.subplot(1,3,2)
plt.imshow(label, cmap="gray")
plt.title("GROUND TRUTH")
plt.axis("off")


plt.subplot(1,3,3)
plt.imshow(rgb_post)
plt.imshow(pred, cmap="autumn", alpha=0.4)
plt.title("PREDICCIÓN")
plt.axis("off")


os.makedirs("results", exist_ok=True)
plt.savefig("results/prediction_example.png", bbox_inches="tight")

plt.show()