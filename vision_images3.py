import rasterio
import numpy as np
import matplotlib.pyplot as plt

dataset_pre3 = rasterio.open("C:/Users/core i7/vision_sentinental/Pareja3_imagenes/subset_S2A_MSIL2A_2019_07_24T000251_N0500_R030_T55HFU_img3_resampled.tif")

print("Número de bandas:", dataset_pre3.count)

B2_pre3 = dataset_pre3.read(1).astype(float)
B3_pre3 = dataset_pre3.read(2).astype(float)
B4_pre3 = dataset_pre3.read(3).astype(float)
B8_pre3 = dataset_pre3.read(4).astype(float)
B11_pre3 = dataset_pre3.read(5).astype(float)
B12_pre3 = dataset_pre3.read(6).astype(float)

B2_pre3 /= 10000
B3_pre3 /= 10000
B4_pre3 /= 10000
B8_pre3 /= 10000
B11_pre3 /= 10000
B12_pre3 /= 10000

#Índices espectrales
ndvi_pre3 = (B8_pre3 - B4_pre3) / (B8_pre3 + B4_pre3 + 1e-10)
nbr_pre3 = (B8_pre3 - B12_pre3) / (B8_pre3 + B12_pre3 + 1e-10)

plt.imshow(nbr_pre3, cmap='RdYlGn')
plt.title("NBR (Pre-incendio)")
plt.colorbar()
plt.show()

dataset_pos3 = rasterio.open("C:/Users/core i7/vision_sentinental/Pareja3_imagenes/subset_S2A_MSIL2A_2020-07-18T000251_N0500_R030_T55HFU_img3_resampled.tif")

print("Número de bandas:", dataset_pos3.count)

B2_pos3 = dataset_pos3.read(1).astype(float)
B3_pos3 = dataset_pos3.read(2).astype(float)
B4_pos3 = dataset_pos3.read(3).astype(float)
B8_pos3 = dataset_pos3.read(4).astype(float)
B11_pos3 = dataset_pos3.read(5).astype(float)
B12_pos3 = dataset_pos3.read(6).astype(float)

B2_pos3 /= 10000
B3_pos3 /= 10000
B4_pos3 /= 10000
B8_pos3 /= 10000
B11_pos3 /= 10000
B12_pos3 /= 10000

#Índices espectrales
ndvi_pos3 = (B8_pos3 - B4_pos3) / (B8_pos3 + B4_pos3 + 1e-10)
nbr_pos3 = (B8_pos3 - B12_pos3) / (B8_pos3 + B12_pos3 + 1e-10)

plt.imsave("nbr_pre3.png", nbr_pre3[::10, ::10], cmap='RdYlGn', vmin=-1, vmax=1)
plt.imsave("nbr_post3.png", nbr_pos3[::10, ::10], cmap='RdYlGn', vmin=-1, vmax=1)    

plt.imsave("ndvi_pre3.png", ndvi_pre3[::10, ::10], cmap='RdYlGn', vmin=-1, vmax=1)
plt.imsave("ndvi_post3.png", ndvi_pos3[::10, ::10], cmap='RdYlGn', vmin=-1, vmax=1)   

import os

os.makedirs("bandas", exist_ok=True)

def guardar_banda(banda, nombre):
    banda = banda.astype(np.float32)
    
    # normalizar entre 0 y 1
    banda_norm = (banda - banda.min()) / (banda.max() - banda.min() + 1e-10)
    
    # reducir tamaño para no romper RAM
    banda_small = banda_norm[::10, ::10]
    
    plt.imsave(f"bandas_pre3/{nombre}.png", banda_small, cmap='gray')

guardar_banda(B2_pre3, "B2_pre3")
guardar_banda(B3_pre3, "B3_pre3")
guardar_banda(B4_pre3, "B4_pre3")
guardar_banda(B8_pre3, "B8_pre3")
guardar_banda(B11_pre3, "B11_pre3")
guardar_banda(B12_pre3, "B12_pre3")

dNDVI3 = ndvi_pre3 - ndvi_pos3

plt.imshow(dNDVI3[::10, ::10], cmap='RdYlGn', vmin=-1, vmax=1)
plt.title("dNDVI (Cambio en vegetación)")
plt.colorbar()
plt.show()

dnbr3 = nbr_pre3 - nbr_pos3

plt.imshow(dnbr3[::10, ::10], cmap='RdBu', vmin=-1, vmax=1)
plt.title("dNBR (Severidad de incendio)")
plt.colorbar()
plt.show()

plt.imsave("dndvi3.png", dNDVI3[::10, ::10], cmap='RdYlGn', vmin=-1, vmax=1)
plt.imsave("dnbr3.png", dnbr3[::10, ::10], cmap='RdBu', vmin=-1, vmax=1)

# -------------------------------------------------------------
# -------------------------------------------------------------

from rasterio.windows import Window

window = Window(3000, 3000, 2048, 2048)

B2 = dataset_pos3.read(1, window=window).astype(float) /10000
B3 = dataset_pos3.read(2, window=window).astype(float) /10000
B4 = dataset_pos3.read(3, window=window).astype(float) /10000
B8 = dataset_pos3.read(4, window=window).astype(float) /10000
B11 = dataset_pos3.read(5, window=window).astype(float) /10000
B12 = dataset_pos3.read(6, window=window).astype(float) /10000

NDVI = (B8 - B4) / (B8 + B4 + 1e-10)
NBR  = (B8 - B12) / (B8 + B12 + 1e-10)
NDMI = (B8 - B11) / (B8 + B11 + 1e-10)

stack_pos3 = np.stack([
    B2, B3, B4, B8, B11, B12,
    NDVI, NBR, NDMI
], axis=0)

np.save("stack_pos3.npy", stack_pos3)