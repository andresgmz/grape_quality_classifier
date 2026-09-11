# Report — Fresh vs. Rotten Grape Detection

Group:
Fruit: grape
Date:

## 1. Dataset

Fuente: **GrapeNet — An Image Dataset for Grape Variety and Quality Classification**,
subconjunto *black round* (ver `data/README.md`).

| | Archivos | Fotos originales |
| --- | ---: | ---: |
| `round_fresh` (label 1) | 2.900 | 116 |
| `round_rotten` (label 0) | 3.000 | 120 |
| **Total** | **5.900** | **236** |

El hallazgo más importante de la exploración: **el dataset no contiene ni una sola
imagen original**. Los 5.900 archivos son 25 variantes aumentadas (5 transformaciones
— `brightness_contrast`, `gamma`, `horizontal_flip`, `rgb_shift`, `rotate` — × 5
iteraciones) de apenas **236 fotografías distintas**. El tamaño real del dataset es
236, no 5.900.

De ahí dos decisiones:

1. **Submuestreo a 4 variantes por foto → 944 imágenes.** Se conservan las 236 fotos
   (que es donde está la información real) y se recorta la redundancia sintética. La
   variación restante la genera nuestra propia capa de augmentation al entrenar.
2. **Split agrupado por foto original.** Un split aleatorio por archivo pondría
   variantes de la misma foto en train y en test, y la precisión reportada sería
   falsa. `tests/test_scaffolding.py` verifica que ninguna foto cruce de conjunto.

| Split | Imágenes | Fotos | rotten / fresh (fotos) |
| --- | ---: | ---: | --- |
| train | 568 | 142 | 72 / 70 |
| val | 188 | 47 | 24 / 23 |
| test | 188 | 47 | 24 / 23 |

## 2. Preprocessing

- Redimensionado a **128×128** y normalización a `[0, 1]` (`rescale 1/255`).
- Augmentation solo en entrenamiento: flip horizontal, rotación ±10%, zoom ±10%.
- Los parámetros se guardan en `models/preprocessing.json` junto al modelo, para
  aplicar en inferencia exactamente la misma transformación que en entrenamiento.

## 3. Model

CNN secuencial, **27.809 parámetros**:

```
3 × [Conv2D(16/32/64, 3×3, relu) + MaxPooling2D]
GlobalAveragePooling2D -> Dropout(0.3) -> Dense(64, relu) -> Dense(1, sigmoid)
```

Optimizador Adam (`lr=1e-3`), pérdida `binary_crossentropy`, batch 16, semilla 42.
Tres bloques convolucionales alcanzan para 128×128 y un dataset de este tamaño; más
capas solo aumentarían el sobreajuste. `EarlyStopping(patience=5)` sobre
`val_accuracy`: se detuvo en la época 22 y restauró los pesos de la **época 17**.

## 4. Results

| Métrica | Validación | Test |
| --- | ---: | ---: |
| Accuracy (por imagen) | 0,926 | **0,872** |
| Accuracy (por foto, voto entre variantes) | 0,957 | **0,915** |
| Precision | 0,898 | 0,809 |
| Recall | 0,957 | 0,967 |
| F1 | 0,926 | 0,881 |

Matriz de confusión en test (filas = real, columnas = predicho):

| | rotten | fresh |
| --- | ---: | ---: |
| **rotten** | 75 | 21 |
| **fresh** | 3 | 89 |

Gráficas: `confusion_matrix.png`, `training_curves.png`. Métricas crudas en
`metrics.json`.

**El error no es simétrico.** El modelo casi nunca se equivoca con la uva fresca
(89/92) pero falla en 21 de 96 imágenes podridas, que clasifica como frescas. Es
decir: recall alto (0,967) a costa de precision (0,809). Para un caso de uso de
control de calidad alimentaria, este es justo el sentido peligroso del error —
dejar pasar fruta podrida es peor que descartar fruta buena.

## 5. Testing with new images

Se copiaron 6 fotos del split de test (nunca vistas en entrenamiento) a
`data/external_test/` y se corrió `python -m src.predict`:

| Imagen | Real | Predicho | Confianza |
| --- | --- | --- | ---: |
| `..._fresh_..._TIMEBURST18_2209_...` | fresh | fresh | 0,874 |
| `..._fresh_..._TIMEBURST7_2121_...` | fresh | fresh | 0,885 |
| `..._fresh_..._TIMEBURST11_2860_...` | fresh | fresh | 0,892 |
| `..._rotten_..._TIMEBURST18_3360_...` | rotten | rotten | 0,689 |
| `..._rotten_..._TIMEBURST19_2876_...` | rotten | rotten | 0,652 |
| `..._rotten_..._TIMEBURST6_2730_...` | rotten | rotten | 0,641 |

**6/6 correctas**, pero la confianza delata el mismo sesgo: las predicciones `fresh`
salen con 0,87–0,89 y las `rotten` apenas con 0,64–0,69.

**Advertencia honesta:** esto **no** es una prueba de generalización real. Estas
imágenes vienen de la misma cámara, la misma sesión y la misma iluminación que el
entrenamiento. Una prueba de verdad necesitaría fotos de otro teléfono, otra luz y
otro fondo.

## 6. Difficulties

- Descubrir que las 5.900 imágenes eran en realidad 236 fotos. Detectarlo tarde
  habría producido un modelo con 99% de accuracy aparente y sin ningún valor.
- Primera corrida con test del 10% (24 fotos): dio 0,646 de accuracy contra 0,883 en
  validación. Con tan pocas fotos la métrica era pura varianza, no señal. Se subió el
  test al 20% (47 fotos).
- `EarlyStopping(restore_best_weights=True)` en TensorFlow 2.11 **solo** restaura los
  mejores pesos si el entrenamiento se corta antes del final. Con `EPOCHS=20` se
  agotaban las épocas sin dispararse y se guardaban los pesos de la última época
  (val 0,904) en vez de los de la mejor (val 0,926). Se subió a `EPOCHS=60`.

## 7. Reflection questions

**¿Qué sesgos podrían tener las imágenes recolectadas?**
Los metadatos reportan una sola cámara (Xiaomi 2312DRAABI) y luz de día en *todas*
las imágenes. Son 236 fotos tomadas en pocas sesiones, sobre el mismo fondo y con el
mismo encuadre. El modelo puede estar aprendiendo condiciones de captura —tono del
fondo, balance de blancos— en vez de la textura de la uva. Además solo hay uva negra
redonda: no sabemos nada sobre uva verde o flame.

**¿Cómo afecta el tamaño del dataset a la precisión?**
El salto de 2 a 4 variantes por foto (472 → 944 imágenes) subió el test de 0,830 a
0,872. Pero el límite real son las 236 fotos: multiplicar variantes sintéticas da
rendimientos decrecientes porque no agrega información nueva. Con 47 fotos en test,
un solo acierto o error mueve la accuracy casi 1 punto porcentual.

**¿Qué limitaciones tiene el modelo entrenado?**
1. Sesgo hacia `fresh`, en el sentido peligroso del error (§4).
2. Solo uva negra redonda, una cámara, una iluminación.
3. Entrenado sobre variantes sintéticas, no sobre fotos genuinamente distintas.
4. Baja confianza en `rotten` (0,64–0,69): el margen de decisión es estrecho y un
   cambio pequeño de umbral alteraría bastante los resultados.

**¿Qué mejoras se podrían hacer en iteraciones futuras?**
Ver §8.

## 8. Possible improvements

1. **Fotografiar uvas propias** con varios teléfonos, fondos e iluminaciones. Es la
   mejora de mayor impacto: ataca la limitación de raíz.
2. **Mover el umbral de decisión** de 0,5 hacia arriba para `fresh`, de modo que la
   duda se resuelva marcando la fruta como podrida — el error menos costoso.
3. **Transfer learning** (MobileNetV2 o similar preentrenado en ImageNet), que es lo
   indicado cuando hay pocos datos reales.
4. **Validación cruzada agrupada** por foto, en vez de un único split: con 47 fotos de
   test, la métrica actual tiene un intervalo de confianza amplio.
5. Incorporar las variedades `flame` y `green` para comprobar si el modelo generaliza
   entre tipos de uva.
