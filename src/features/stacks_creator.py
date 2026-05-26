import numpy as np
import os


def generar_tiles(stack_pre, stack_post, tile_size=512, output_dir="tiles"):
    os.makedirs(output_dir, exist_ok=True)

    _, H, W = stack_pre.shape
    tile_id = 0

    for i in range(0, H, tile_size):
        for j in range(0, W, tile_size):

            tile_pre = stack_pre[:, i:i+tile_size, j:j+tile_size]
            tile_post = stack_post[:, i:i+tile_size, j:j+tile_size]

            if tile_pre.shape[1] == tile_size and tile_pre.shape[2] == tile_size:

                np.save(f"{output_dir}/tile_{tile_id}_pre.npy", tile_pre)
                np.save(f"{output_dir}/tile_{tile_id}_post.npy", tile_post)

                tile_id += 1

    print(f"Total tiles generados: {tile_id}")


def build_all_tiles():
    for i in [1, 2, 3]:
        stack_pre = np.load(f"data/stacks/stack_pre{i}.npy")
        stack_post = np.load(f"data/stacks/stack_pos{i}.npy")

        generar_tiles(
            stack_pre,
            stack_post,
            tile_size=512,
            output_dir=f"data/tiles_par{i}"
        )
