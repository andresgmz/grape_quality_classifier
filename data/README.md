# Data

Las imágenes **no** se versionan. Solo se versionan los CSV de metadatos y este
documento; todo lo que son imágenes está ignorado por git (ver `.gitignore`).

El proyecto pasó por **dos datasets**. El segundo es el que está en uso.

---

## Dataset actual: `grape/`

Carpeta en la raíz del repositorio, con una subcarpeta por clase:

```
grape/fresh/     200 imágenes, label = 1
grape/rotten/    200 imágenes, label = 0
```

Uvas verdes, sueltas o en racimos pequeños, fotografiadas de cerca sobre una
superficie clara. La señal de podredumbre son manchas marrones, hundimientos y piel
arrugada.

### Verificaciones hechas antes de entrenar

| Chequeo | Resultado |
| --- | --- |
| Duplicados exactos (hash MD5) | **0** en ambas clases |
| Tamaños de imagen | Variados, de ~650×650 a 1040×780 |
| Balance de clases | 200 / 200 |
| Brillo medio `fresh` | 147,7 |
| Brillo medio `rotten` | 142,8 |

El brillo importa: al ser casi idéntico entre clases, el modelo **no puede** separarlas
con un atajo de luminosidad y está obligado a mirar la textura. En el dataset anterior
eso no pasaba, y fue exactamente el problema.

### Cómo usarlo

`config.DATASET_DIR` apunta a esta carpeta. Para entrenar con otro dataset, armá una
carpeta con subcarpetas `fresh/` y `rotten/`, cambiá esa línea y volvé a correr
`python -m src.dataset` y `python -m src.train`.

---

## Dataset anterior: GrapeNet (ya no se usa)

**GrapeNet — An Image Dataset for Grape Variety and Quality Classification.**
Subconjunto *black round*, descomprimido en su momento a `data/raw/fresh` y
`data/raw/rotten` por `src/prepare_data.py`.

Se descartó por dos motivos, ambos documentados en la sección 6 del informe.

### 1. No contenía ni una sola imagen original

Los 5.900 archivos eran 25 variantes aumentadas (5 transformaciones —
`brightness_contrast`, `gamma`, `horizontal_flip`, `rgb_shift`, `rotate` — × 5
iteraciones) de solo **236 fotos distintas**: 116 fresh y 120 rotten. El tamaño real
del dataset era 236, no 5.900.

Esto obligaba a **agrupar el split por foto original** (`base_id`, el prefijo
`IMG_<fecha>_<hora>_TIMEBURST<n>_<id>`). Repartir archivos al azar habría puesto
variantes de la misma foto en train y en test, inflando la precisión reportada.
`src/prepare_data.base_image_id()` calcula ese identificador y hay tests que verifican
que ninguna foto cruce de conjunto. **Esa lógica se conserva en el pipeline actual**,
aunque el dataset de hoy no la necesite.

### 2. Las imágenes eran una uva suelta sobre fondo blanco de estudio

Todas las fotos venían de una sola cámara (Xiaomi 2312DRAABI) con luz de día, y
mostraban **una única uva centrada sobre papel blanco**. El fondo ocupaba la mayor
parte del cuadro:

| | Brillo medio (0–255) |
| --- | ---: |
| GrapeNet `fresh` | 244,1 |
| GrapeNet `rotten` | 229,4 |
| Fotos reales de prueba | 121–157 |

El modelo nunca vio una imagen por debajo de 220 de brillo, y la única diferencia
entre clases que tenía a mano era cuántos píxeles oscuros había en el cuadro.
Terminó aprendiendo "oscuro = podrido" en vez de reconocer la podredumbre. Alcanzaba
0,872 en su propio test y fallaba con cualquier foto real.

### Metadatos

`metadata/Black_Round_Fresh.csv` y `metadata/Black_Round_Rotten.csv` vienen del
dataset original y se conservan versionados por trazabilidad. Columnas:

`Grape Category, Image File Name, Type of File, Resolution, Bit Depth, Camera Maker,
Camera Model, DPI, Max Aperture, ISO Speed, Light Source, Flash Mode, Date Created`

---

## Otras carpetas

- `data/processed/labels.csv` — generado por `src/dataset.py`: ruta, etiqueta, clase,
  `base_id` y split de cada imagen. Regenerable, no se versiona.
- `data/external_test/` — imágenes sueltas para probar generalización a mano. Las que
  hay ahora quedaron del dataset anterior.
- `data_to_test/` — fotos descargadas de internet usadas para la prueba de la sección
  6 del informe.
