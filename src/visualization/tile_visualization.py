import numpy as np
import matplotlib.pyplot as plt
import os


tiles_dir = "data/tiles_par3"
tile_id = 0


tile_pre = np.load(os.path.join(tiles_dir, f"tile_{tile_id}_pre.npy"))
tile_post = np.load(os.path.join(tiles_dir, f"tile_{tile_id}_post.npy"))


def to_rgb(tile):

    R = tile[2]
    G = tile[1]
    B = tile[0]

    rgb = np.stack([R, G, B], axis=-1)

    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-10)

    return rgb


rgb_pre = to_rgb(tile_pre)
rgb_post = to_rgb(tile_post)


plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(rgb_pre)
plt.title("PRE")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(rgb_post)
plt.title("POST")
plt.axis("off")

plt.suptitle(f"Tile {tile_id} - RGB (B4, B3, B2)")

os.makedirs("results", exist_ok=True)
plt.savefig("results/tile_example.png", bbox_inches="tight")

plt.show()