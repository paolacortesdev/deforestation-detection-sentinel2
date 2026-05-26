import numpy as np
import os


def generar_labels(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    archivos = sorted([
        f for f in os.listdir(input_dir)
        if "pre" in f
    ])

    for file_pre in archivos:

        file_post = file_pre.replace("pre", "post")

        pre = np.load(os.path.join(input_dir, file_pre))
        post = np.load(os.path.join(input_dir, file_post))

        # índices
        ndvi_pre = pre[6]
        ndvi_post = post[6]

        nbr_pre = pre[7]
        nbr_post = post[7]

        # cambios
        dNDVI = ndvi_post - ndvi_pre
        dNBR = nbr_post - nbr_pre

        # label
        label = np.zeros_like(dNDVI)

        label[dNDVI < -0.2] = 1
        label[dNBR < -0.3] = 1

        out_name = file_pre.replace("pre", "label")

        np.save(os.path.join(output_dir, out_name), label)


def build_all_labels():
    for i in [1, 2, 3]:
        generar_labels(
            f"data/tiles_par{i}",
            f"data/labels_par{i}"
        )