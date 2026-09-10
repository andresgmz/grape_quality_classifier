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

- **The dataset ships pre-augmented.** File names carry suffixes such as
  `_brightness_contrast_iter1` or `_horizontal_flip_iter1`, meaning several
  files derive from the same original photo. A random train/test split would
  leak augmented variants of the same base image across sets and inflate the
  reported accuracy. Splits must be **grouped by base image** (the
  `IMG_<date>_<time>_TIMEBURST<n>_<id>` prefix).
- **Single acquisition setup.** The metadata reports one camera maker/model
  (Xiaomi 2312DRAABI) and daylight as the light source for every image, so the
  model may not generalize to other cameras or lighting conditions. Relevant to
  the bias question in the assignment.
