# Grape Quality Classifier

CNN image classifier that detects fresh vs. rotten grapes.

Course project for **Advanced Topics in Software Engineering** (Master's, semester 1).
Fruit chosen by the group: **grape**. Binary task: `0 = rotten`, `1 = fresh`.

Arquitectura y diagramas: [`ARQUITECTURA.md`](ARQUITECTURA.md).

Pipeline completo y entrenado. Resultado en el conjunto de test:
**0,872 de accuracy por imagen** y **0,915 por foto original**. Detalles, matriz de
confusión y reflexiones en [`reports/report.md`](reports/report.md).

## Structure

```
data/
  raw/fresh/       original grape images, label = 1
  raw/rotten/      original grape images, label = 0
  processed/       train-ready dataset (splits, labels.csv)
  external_test/   new images to check generalization
  metadata/        GrapeNet metadata CSVs (versioned)
  README.md        dataset source, layout and caveats
models/            trained model + preprocessing artifacts
notebooks/         exploration
reports/           results report
src/               source code
tests/             tests
```

## Usage

```bash
# entorno aislado (Python 3.9)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate      # Linux / macOS
pip install -r requirements.txt

python -m src.prepare_data [ruta_a_Grapes_Dataset]   # descomprime el subset black en data/raw/
python -m src.dataset                                # arma labels.csv y los splits
python -m src.train                                  # entrena y guarda modelo + preprocessing
python -m src.evaluate                               # metricas, matriz de confusion y curvas
python -m src.predict data/external_test/*.jpg       # inferencia sobre imagenes nuevas

python -m pytest tests/                              # tests del pipeline de datos
```

`prepare_data` busca el archivo en `~/Downloads/GrapeNet.../Grapes_Dataset` por
defecto; se le puede pasar otra ruta como argumento.

Las versiones de `requirements.txt` están fijadas a las que produjeron los
resultados del informe. TensorFlow 2.11 es la última serie que funciona con
Python 3.9 sin pasar a Keras 3, que no carga el modelo `.keras` guardado aquí.

## Decisiones de diseño

- **El dataset trae 5.900 archivos pero solo 236 fotos distintas**: cada foto viene
  con 25 variantes aumentadas. Usamos 4 variantes por foto (944 imágenes) y
  generamos el resto de la variación con augmentation propia al entrenar.
- **El split se hace por foto original, no por archivo.** Repartir archivos al azar
  pondría variantes de la misma foto en train y test, e inflaría la precisión. Hay
  un test que lo verifica.
- El preprocessing se serializa en `models/preprocessing.json` junto al modelo, para
  que inferencia y entrenamiento apliquen la misma transformación.

## Estado

- [x] Extract the GrapeNet black grape subset (see `data/README.md`)
- [x] Implement preprocessing and data augmentation
- [x] Define and train the CNN
- [x] Evaluate and test with new images
- [x] Save model and preprocessing artifacts
- [x] Write the report (`reports/report.md`)
