# Report — Fresh vs. Rotten Grape Detection

Group:
Fruit: grape
Date:

> **Resumen.** Clasificador binario de uvas fresh/rotten con MobileNetV2 y transfer
> learning. **Accuracy 0,983 en test**, con **0 uvas podridas clasificadas como
> frescas** gracias a un umbral de decisión calibrado en 0,62. El proyecto pasó por
> dos iteraciones: la primera falló por un problema de dominio que vale la pena
> contar, porque es el aprendizaje principal del trabajo.

---

## 0. Las dos iteraciones

| | Iteración 1 | Iteración 2 (final) |
| --- | --- | --- |
| Dataset | GrapeNet, subset black round | `grape/` |
| Imágenes | 5.900 archivos / **236 fotos reales** | **400 imágenes únicas** |
| Contenido | 1 uva suelta sobre fondo blanco | Uvas y racimos pequeños sobre superficie lisa |
| Modelo | CNN propia, 27.809 parámetros | **MobileNetV2 preentrenada** |
| Split | 60 / 20 / 20 | **70 / 15 / 15** |
| Umbral | 0,5 fijo | **0,62 calibrado** |
| Accuracy test | 0,872 | **0,983** |
| Podridas que escapan | 21 de 96 | **0 de 30** |

La iteración 1 alcanzaba 0,872 en su propio test, pero al probarla con fotos reales
descubrimos que el número no significaba lo que parecía. La sección 6 lo explica.

---

## 1. Dataset

**Iteración final:** carpeta `grape/`, con una subcarpeta por clase.

| | Imágenes | Etiqueta |
| --- | ---: | ---: |
| `grape/fresh/` | 200 | 1 |
| `grape/rotten/` | 200 | 0 |
| **Total** | **400** | |

Verificado antes de entrenar:

- **400 imágenes únicas**, sin duplicados exactos (hash MD5 de cada archivo).
- **Tamaños variados** (de ~650×650 a 1040×780), señal de que no vienen de una sola
  sesión de captura automatizada.
- **Brillo medio prácticamente igual entre clases**: 147,7 (fresh) contra 142,8
  (rotten). Esto importa: significa que el modelo **no puede** separar las clases con
  un atajo de luminosidad y está obligado a mirar la textura y las manchas.

Contenido: uvas verdes, sueltas o en racimos pequeños, fotografiadas de cerca sobre
una superficie clara. La señal de podredumbre son manchas marrones, hundimientos y
piel arrugada.

### Split 70 / 15 / 15

| Split | Imágenes | rotten / fresh |
| --- | ---: | --- |
| train | 280 | 140 / 140 |
| val | 60 | 30 / 30 |
| test | 60 | 30 / 30 |

Perfectamente balanceado, así que la accuracy es interpretable directamente (una
línea base trivial que responda siempre lo mismo daría 0,500).

El reparto se hace agrupando por `base_id`. En este dataset cada archivo es una
imagen distinta, así que el agrupamiento coincide con un split estratificado normal;
la lógica se conserva porque protege el caso de un dataset con variantes de una
misma foto, que es exactamente lo que nos mordió en la iteración 1.

---

## 2. Preprocessing

- Redimensionado a **128×128** y normalización a `[0, 1]`.
- Augmentation **solo en entrenamiento**: flip horizontal, rotación ±10%, zoom ±10%.
- Una capa `Rescaling(2, offset=-1)` convierte `[0, 1]` a `[-1, 1]`, que es lo que
  espera MobileNetV2.

Los parámetros se serializan en `models/preprocessing.json` junto al modelo, de modo
que la inferencia aplica exactamente la misma transformación que el entrenamiento.
Ese archivo también guarda el umbral de decisión (sección 4).

---

## 3. Model

**MobileNetV2 preentrenada en ImageNet, con la base congelada.**

```
Input 128×128×3
  -> Augmentation (solo train)
  -> Rescaling [0,1] -> [-1,1]
  -> MobileNetV2 (congelada, include_top=False)
  -> GlobalAveragePooling2D -> Dropout(0.3)
  -> Dense(1, sigmoid)
```

| | |
| --- | ---: |
| Parámetros totales | 2.259.265 |
| Parámetros entrenables | **1.281** (0,1%) |

Solo se entrena la cabeza. Con 280 imágenes de entrenamiento, ajustar 2,2 millones de
pesos sería sobreajustar garantizado; en cambio unas features ya aprendidas sobre
millones de fotos se reutilizan tal cual y solo hace falta aprender a combinarlas.

Adam con `lr = 1e-4` (más bajo que el `1e-3` de la red desde cero: la cabeza sobre una
base congelada no necesita avanzar rápido), batch 16, semilla 42.

`EarlyStopping` sobre **`val_loss`** con `patience=8`. Al principio monitoreaba
`val_accuracy`, pero con 60 imágenes de validación la accuracy se mueve a saltos de
1,7% y frenaba el entrenamiento mientras la pérdida seguía bajando. Cambiar la señal
subió la accuracy de validación de 0,933 a 0,967.

La CNN propia de la iteración 1 sigue disponible con `config.MODEL_KIND = "cnn"` como
línea base de comparación.

---

## 4. El umbral de decisión

**No todos los errores cuestan lo mismo.** Dejar pasar una uva podrida como fresca es
grave; descartar una sana solo cuesta fruta. Un umbral de 0,5 trata ambos errores
como equivalentes, lo cual es una decisión implícita y equivocada.

`train.py` barre umbrales de 0,30 a 0,95 **sobre el conjunto de validación** y elige
el más bajo que no deje escapar ninguna podrida, desempatando por accuracy. El barrido:

| Umbral | Podridas detectadas | Frescas correctas | **Escapan** | Accuracy val |
| ---: | ---: | ---: | ---: | ---: |
| 0,50 | 29/30 | 29/30 | **1** | 0,967 |
| **0,62** | **30/30** | 28/30 | **0** | **0,967** |
| 0,80 | 30/30 | 25/30 | 0 | 0,917 |
| 0,95 | 30/30 | 17/30 | 0 | 0,783 |

**0,62 elimina el error caro sin costar nada de accuracy.** Por encima se empieza a
descartar fruta sana sin ganar nada.

El umbral se guarda en `models/preprocessing.json`, así que `evaluate` y `predict`
usan el mismo valor que se calibró. Si se cambia de dataset y se reentrena, se
recalibra solo.

---

## 5. Results

| Métrica | Validación | Test |
| --- | ---: | ---: |
| Accuracy | 0,967 | **0,983** |
| Precision | 1,000 | **1,000** |
| Recall (fresh) | 0,933 | 0,967 |
| F1 | 0,966 | 0,983 |
| **Podridas detectadas** | **1,000** | **1,000** |

Matriz de confusión en test (filas = real, columnas = predicho):

| | rotten | fresh |
| --- | ---: | ---: |
| **rotten** | **30** | **0** |
| **fresh** | 1 | 29 |

Los dos números que importan:

- **Ninguna uva podrida pasó como fresca** (0 de 30). Es el error que queríamos
  eliminar.
- **Precision 1,000**: cuando el modelo dice "fresh", acierta siempre. El único error
  es una uva sana descartada de más, que es el error barato.

Comparado con el umbral 0,5, la accuracy es la misma (0,983) pero el error cambió de
lado. Eso es una mejora aunque el número global no se mueva.

Gráficas: `confusion_matrix.png`, `training_curves.png`. Métricas crudas en
`metrics.json`.

---

## 6. Testing with new images

Se probó con 4 fotos descargadas de internet, ajenas por completo a los datasets:
un racimo de uvas negras frescas sobre fondo blanco y tres de racimos podridos en
viñedo (uno sostenido por una mano, otro con botritis, otro con uvas secas).

**Con el modelo de la iteración 1 (GrapeNet): 2 de 4, y por el motivo equivocado.**
El racimo fresco fue clasificado como podrido con 0,839 de confianza. Investigando:

| | Brillo medio (0–255) |
| --- | ---: |
| Entrenamiento GrapeNet `fresh` | 244,1 |
| Entrenamiento GrapeNet `rotten` | 229,4 |
| Las 4 fotos de prueba | 121–157 |

Las imágenes de GrapeNet son **una sola uva sobre fondo blanco de estudio**: el fondo
ocupa la mayor parte del cuadro y por eso el brillo medio ronda 240. El modelo nunca
vio una imagen por debajo de 220, y la única diferencia entre clases que tenía a mano
era cuántos píxeles oscuros había. Aprendió "oscuro = podrido". Un racimo de uvas
negras brillantes está lleno de píxeles oscuros, así que lo llamó podrido con
confianza. Acertó dos de cuatro por casualidad, porque eran oscuras.

**Con el modelo final: 4 de 4.**

| Imagen | Real | `p_fresh` | Umbral | Predicho |
| --- | --- | ---: | ---: | --- |
| Racimo fresco, fondo blanco | fresh | 0,903 | 0,62 | fresh |
| Racimo podrido en mano | rotten | 0,521 | 0,62 | rotten |
| Uvas secas en la vid | rotten | 0,570 | 0,62 | rotten |
| Racimo con botritis | rotten | 0,515 | 0,62 | rotten |

**Pero hay que leer las probabilidades, no las etiquetas.** Las tres podridas están
entre 0,515 y 0,570: el modelo sigue sin entenderlas, está en la duda. Lo que las
clasifica bien es la política conservadora del umbral —"ante la duda, podrida"— y da
la casualidad de que tres de las cuatro son podridas. Con cuatro racimos **frescos**
de viñedo, el modelo los llamaría podridos a todos.

El 4/4 es el umbral funcionando, no el modelo generalizando. La brecha de dominio
sigue ahí: `grape/` son uvas verdes sobre superficie lisa y estas son racimos negros
en viñedo. Lo único que la cierra son datos de ese tipo.

Nótese, además, la mejora cualitativa: el modelo viejo se equivocaba **con
confianza** (0,839); el nuevo se equivoca **dudando** (0,515). Un modelo que sabe que
no sabe es mucho más útil, porque permite derivar los casos dudosos a revisión humana.

---

## 7. Difficulties

1. **El dataset de la iteración 1 no tenía ni una imagen original.** Los 5.900
   archivos eran 25 variantes aumentadas de 236 fotos. Un split aleatorio por archivo
   habría puesto variantes de la misma foto en train y test, con una accuracy
   aparente altísima y sin valor. Se resolvió agrupando el split por foto original.
2. **Test demasiado chico.** Con 10% de test sobre 236 fotos quedaban 24 imágenes y
   la métrica era pura varianza: 0,646 en test contra 0,883 en validación. Se amplió
   el test.
3. **`EarlyStopping` que no se disparaba.** En TensorFlow 2.11,
   `restore_best_weights=True` solo restaura los mejores pesos si el entrenamiento se
   corta antes de agotar las épocas. Con el tope de épocas demasiado bajo se guardaban
   los pesos de la última época en vez de los de la mejor.
4. **Monitorear la métrica equivocada.** `val_accuracy` sobre 60 imágenes es una señal
   demasiado gruesa; cambiar a `val_loss` subió la validación de 0,933 a 0,967.
5. **Entrenar de más también perjudica.** Se probó con 200 épocas: la `val_loss`
   siguió bajando pero la accuracy de validación cayó a 0,950 y el train llegó a
   0,989 — sobreajuste. Se volvió a 60.
6. **El fallo de dominio de la sección 6**, que obligó a cambiar de dataset. Fue la
   dificultad más costosa y la más instructiva.

---

## 8. Reflection questions

**¿Qué sesgos podrían tener las imágenes recolectadas?**
Las 400 imágenes son de uvas **verdes** sobre superficies claras y lisas, fotografiadas
de cerca y desde arriba. No hay uva negra ni roja, ni fondos complejos, ni fruta en la
planta, ni luz artificial. El modelo hereda ese recorte: funciona en ese escenario y
se pierde fuera de él, como demostró la sección 6. La iteración 1 tenía un sesgo aún
más severo —una sola cámara y una sola sesión— que lo llevó a aprender el brillo del
fondo en vez de la fruta.

**¿Cómo afecta el tamaño del dataset a la precisión?**
Con 280 imágenes de entrenamiento, entrenar desde cero no alcanza: la CNN propia de
27.809 parámetros necesitaba muchos más datos. El transfer learning resuelve
justamente eso —reutiliza features aprendidas sobre millones de imágenes y solo
entrena 1.281 parámetros— y por eso funciona con tan poco. O sea: con pocos datos, la
arquitectura importa más que la cantidad. Aun así, 60 imágenes de test siguen siendo
pocas: un solo error mueve la accuracy 1,7 puntos, y el 0,983 tiene un intervalo de
confianza ancho.

**¿Qué limitaciones tiene el modelo entrenado?**
1. Solo uva verde sobre fondo liso; falla con racimos en viñedo (sección 6).
2. El test son 60 imágenes: la métrica es orientativa, no definitiva.
3. La base está congelada, así que el modelo nunca adaptó sus features a la textura
   específica de la uva podrida.
4. El umbral de 0,62 está calibrado para **este** dataset. Con otra distribución de
   imágenes hay que recalibrarlo.

**¿Qué mejoras se podrían hacer en iteraciones futuras?**
Ver sección 9.

---

## 9. Possible improvements

1. **Fine-tuning de las últimas capas de MobileNetV2**, con learning rate muy bajo
   tras entrenar la cabeza. Es el siguiente paso natural y suele dar varios puntos.
2. **Ampliar el dataset al escenario real de uso**: uva negra y roja, fruta en la
   planta, fondos e iluminaciones variadas. Es lo único que cierra la brecha de la
   sección 6.
3. **Zona de rechazo en vez de un único umbral**: por debajo de 0,62 marcar "podrida",
   por encima de 0,85 "fresca", y derivar el medio a revisión humana. Los casos
   dudosos de la sección 6 caían justo ahí.
4. **Validación cruzada estratificada** en lugar de un único split, para acotar la
   incertidumbre de la métrica.
5. **Ponderar las clases en la función de pérdida**, para que el costo asimétrico del
   error entre en el entrenamiento y no solo en el umbral posterior.
