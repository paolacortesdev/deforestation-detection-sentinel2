# deforestation-detection-sentinel2
## Resultados preliminares

Se han generado mapas de índices espectrales como NDVI y NBR, así como diferencias multitemporales (dNDVI y dNBR) que permiten identificar cambios en la vegetación.

## Estructura del proyecto

- `src/preprocessing`: carga de imágenes Sentinel-2, normalización de bandas y cálculo de índices espectrales (NDVI, NBR, dNDVI, dNBR)

- `src/features`: generación de stacks multicanal, división en tiles (512x512) y creación de etiquetas a partir de umbrales sobre índices espectrales

- `src/visualization`: visualización de tiles en formato RGB (B4, B3, B2) para comparación entre imágenes pre y post evento

## Pipeline de datos

Imagen Sentinel-2 → Preprocesamiento → Cálculo de índices → Stack multicanal → Tiles → Labels → (Modelo en desarrollo)

## Ejemplo de resultados

Comparación entre imagen pre-evento y post-evento en formato RGB:

![Ejemplo de tile](results/tile_example.png)

## Próximos pasos

- Implementación de modelo U-Net  
- Entrenamiento y evaluación  