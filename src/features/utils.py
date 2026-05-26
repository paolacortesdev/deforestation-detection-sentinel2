import os


def get_tile_ids(tiles_dir):
    return sorted([
        f.split("_")[1]
        for f in os.listdir(tiles_dir)
        if "pre" in f
    ])


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)