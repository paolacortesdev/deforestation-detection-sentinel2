import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

# =====================================================
# DEVICE
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Usando dispositivo: {device}")

# =====================================================
# CARGAR MODELO
# =====================================================

model = smp.DeepLabV3Plus(
    encoder_name="resnet50",
    encoder_weights=None,
    in_channels=18,
    classes=1
)

model.load_state_dict(
    torch.load(
        "models/best_deeplabv3plus.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("Modelo cargado correctamente")

# =====================================================
# CARGAR TILE ALEATORIO
# =====================================================

tiles_dir = "tiles_demo"

tiles = sorted([
    f for f in os.listdir(tiles_dir)
    if "pre" in f
])

tile_random = random.choice(tiles)

print(f"Tile seleccionado: {tile_random}")

# PRE
pre = np.load(
    os.path.join(
        tiles_dir,
        tile_random
    )
)

# POST
post_name = tile_random.replace("pre", "post")

post = np.load(
    os.path.join(
        tiles_dir,
        post_name
    )
)

print("PRE shape :", pre.shape)
print("POST shape:", post.shape)

# =====================================================
# PREPARAR INPUT
# =====================================================

x = np.concatenate([pre, post], axis=0)

x = torch.tensor(
    x,
    dtype=torch.float32
)

x = x.unsqueeze(0).to(device)

print("Input final:", x.shape)

# =====================================================
# INFERENCIA
# =====================================================

with torch.no_grad():

    pred = model(x)

    pred = torch.sigmoid(pred)

    # threshold
    pred_bin = (pred > 0.2).float()

pred_bin = pred_bin.cpu().numpy()[0,0]

# =====================================================
# RGB VISUAL
# =====================================================

rgb_pre = np.transpose(
    pre[[2,1,0]],
    (1,2,0)
)

rgb_post = np.transpose(
    post[[2,1,0]],
    (1,2,0)
)

# normalizar
rgb_pre = (
    rgb_pre - rgb_pre.min()
) / (
    rgb_pre.max() - rgb_pre.min()
)

rgb_post = (
    rgb_post - rgb_post.min()
) / (
    rgb_post.max() - rgb_post.min()
)

# =====================================================
# MÉTRICAS
# =====================================================

pixeles_deforestados = np.sum(pred_bin)

total_pixeles = (
    pred_bin.shape[0] *
    pred_bin.shape[1]
)

porcentaje = (
    pixeles_deforestados /
    total_pixeles
) * 100

print(f"\nÁrea afectada: {porcentaje:.2f}%")

# =====================================================
# VISUALIZACIÓN
# =====================================================

plt.figure(figsize=(18,6))

# PRE
ax1 = plt.subplot(1,3,1)

ax1.imshow(rgb_pre)

ax1.set_title(
    "ANTES (PRE)",
    fontsize=18,
    fontweight="bold"
)

ax1.axis("off")

# POST
ax2 = plt.subplot(1,3,2)

ax2.imshow(rgb_post)

ax2.set_title(
    "DESPUÉS (POST)",
    fontsize=18,
    fontweight="bold"
)

ax2.axis("off")

# PREDICCIÓN
ax3 = plt.subplot(1,3,3)

ax3.imshow(rgb_post)

mask = np.ma.masked_where(
    pred_bin == 0,
    pred_bin
)

ax3.imshow(
    mask,
    cmap="autumn",
    alpha=0.4
)

ax3.set_title(
    "DEFORESTACIÓN DETECTADA",
    fontsize=18,
    fontweight="bold"
)

ax3.axis("off")

plt.suptitle(
    "Sistema de Detección de Deforestación",
    fontsize=24,
    fontweight="bold"
)

plt.figtext(
    0.5,
    -0.05,
    f"Área afectada: {porcentaje:.2f}%",
    ha="center",
    fontsize=16
)

plt.tight_layout()

plt.show()