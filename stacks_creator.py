import numpy as np
import os

def generar_tiles(stack_pre, stack_post, tile_size=512, output_dir="tiles"):
    import os
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

for i in [1, 2, 3]:
    stack_pre = np.load(f"stack_pre{i}.npy")
    stack_post = np.load(f"stack_pos{i}.npy")
    
    generar_tiles(
        stack_pre,
        stack_post,
        tile_size=512,
        output_dir=f"tiles_par{i}"
    )

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------

import numpy as np
import os

def generar_labels(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    archivos = sorted([f for f in os.listdir(input_dir) if "pre" in f])
    
    for file_pre in archivos:
        file_post = file_pre.replace("pre", "post")
        
        pre = np.load(os.path.join(input_dir, file_pre))
        post = np.load(os.path.join(input_dir, file_post))
        
        # Extraer índices
        ndvi_pre = pre[6]
        ndvi_post = post[6]
        
        nbr_pre = pre[7]
        nbr_post = post[7]
        
        # Cambios
        dNDVI = ndvi_post - ndvi_pre
        dNBR = nbr_post - nbr_pre
        
        # Regla simple (puedes ajustar)
        label = np.zeros_like(dNDVI)
        
        # Deforestación (NDVI baja)
        label[dNDVI < -0.2] = 1
        
        # Incendio (NBR baja fuerte)
        label[dNBR < -0.3] = 1
        
        # Guardar
        np.save(os.path.join(output_dir, file_pre.replace("pre", "label")), label)

    print("Labels generados")

generar_labels("tiles_par1", "labels_par1")
generar_labels("tiles_par2", "labels_par2")
generar_labels("tiles_par3", "labels_par3")

import numpy as np
import matplotlib.pyplot as plt

label = np.load(r"C:\Users\core i7\vision_sentinental\labels_par1\tile_0_label.npy")

plt.imshow(label, cmap="gray")
plt.title("Label")
plt.colorbar()
plt.show()
