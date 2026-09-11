# Data

Images are **not** tracked in this repository. Only the metadata CSVs and this
document are versioned; everything under `raw/`, `processed/` and
`external_test/` is ignored by git (see `.gitignore`).

Reason: the black-grape subset alone weighs ~895 MB compressed, which is far
above GitHub's 100 MB per-file limit and would bloat the history permanently.
The dataset is therefore reproduced locally from the original source.

## Source

**GrapeNet — An Image Dataset for Grape Variety and Quality Classification.**

The full dataset covers three varieties (`black`, `flame`, `green`) split into
`fresh` / `rotten` folders, each one distributed as a ZIP file. This project
uses **only the black round subset**:

| Subset                              | Size    | Label |
| ----------------------------------- | ------- | ----- |
| `Grapes_Dataset/black/round_fresh`  | ~359 MB | 1     |
| `Grapes_Dataset/black/round_rotten` | ~535 MB | 0     |

## How to get the data

1. Download the GrapeNet archive from its original source.
2. Extract `Grapes_Dataset/black/round_fresh/black_round_fresh.zip` and
   `Grapes_Dataset/black/round_rotten/black_round_rotten.zip`.
3. Place the images so the layout is:

```
data/raw/fresh/    <- images from black_round_fresh.zip   (label = 1)
data/raw/rotten/   <- images from black_round_rotten.zip  (label = 0)
```

`python -m src.prepare_data` is meant to automate steps 2 and 3 (not
implemented yet).

## Metadata

`metadata/Black_Round_Fresh.csv` and `metadata/Black_Round_Rotten.csv` come
from the original dataset and are small enough to version. Columns:

`Grape Category, Image File Name, Type of File, Resolution, Bit Depth,
Camera Maker, Camera Model, DPI, Max Aperture, ISO Speed, Light Source,
Flash Mode, Date Created`

## Known caveats

- **No hay ni una sola imagen original.** Los 5.900 archivos son 25 variantes
  aumentadas (5 transformaciones — `brightness_contrast`, `gamma`,
  `horizontal_flip`, `rgb_shift`, `rotate` — × 5 iteraciones) de solo **236 fotos
  distintas**: 116 fresh y 120 rotten. El tamaño real del dataset es 236, no 5.900.
- **El split debe agruparse por foto original** (`base_id`, el prefijo
  `IMG_<date>_<time>_TIMEBURST<n>_<id>`). Repartir archivos al azar pondría
  variantes de la misma foto en train y en test e inflaría la precisión reportada.
  `src/prepare_data.base_image_id()` calcula ese identificador y hay tests que
  verifican que ninguna foto cruce de conjunto.
- **Se usan 4 variantes por foto** (944 imágenes, `config.VARIANTS_PER_PHOTO`), no
  las 25: se conservan todas las fotos distintas y se recorta la redundancia
  sintética, que la augmentation propia del entrenamiento ya genera.
- **Single acquisition setup.** The metadata reports one camera maker/model
  (Xiaomi 2312DRAABI) and daylight as the light source for every image, so the
  model may not generalize to other cameras or lighting conditions. Relevant to
  the bias question in the assignment.
