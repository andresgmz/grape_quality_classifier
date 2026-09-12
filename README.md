# Grape Quality Classifier

Clasificador de imágenes que detecta uvas **frescas vs. podridas**, con transfer
learning sobre MobileNetV2.

Proyecto de **Tópicos Avanzados de Ingeniería de Software** (maestría, semestre 1).
Fruta elegida por el grupo: **uva**. Tarea binaria: `0 = rotten`, `1 = fresh`.

| | Test |
| --- | ---: |
| Accuracy | **0,983** |
| Precision | **1,000** |
| Podridas detectadas | **1,000** |
| Podridas que pasan como frescas | **0 de 30** |

Arquitectura y diagramas: [`ARQUITECTURA.md`](ARQUITECTURA.md).
Informe completo: [`reports/report.md`](reports/report.md).

## Estructura

```
grape/
  fresh/           imágenes de uva sana, label = 1
  rotten/          imágenes de uva podrida, label = 0
data/
  processed/       labels.csv con los splits
  external_test/   imágenes para probar generalización
  metadata/        CSVs de GrapeNet (dataset anterior)
  README.md        datasets, procedencia y caveats
models/            modelo entrenado + preprocessing.json
notebooks/         exploración
reports/           informe, métricas y gráficas
src/               código
tests/             tests
```

## Usage

```bash
# entorno aislado (Python 3.9)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate      # Linux / macOS
pip install -r requirements.txt

python -m src.dataset                          # arma labels.csv y los splits 70/15/15
python -m src.train                            # entrena, calibra el umbral y guarda artefactos
python -m src.evaluate                         # métricas, matriz de confusión y curvas
python -m src.predict ruta/a/imagen.jpg        # inferencia sobre imágenes nuevas

python -m pytest tests/                        # tests del pipeline de datos
```

`python -m src.prepare_data` pertenece al dataset anterior (descomprimía el archivo de
GrapeNet) y ya no hace falta en el flujo actual.

Las versiones de `requirements.txt` están fijadas a las que produjeron los resultados
del informe. TensorFlow 2.11 es la última serie que funciona con Python 3.9 sin pasar
a Keras 3, que no carga el modelo `.keras` guardado aquí.

## Decisiones de diseño

- **Transfer learning sobre MobileNetV2 congelada.** Solo se entrenan 1.281 de
  2.259.265 parámetros. Con 280 imágenes de entrenamiento, ajustar la red completa
  sería sobreajustar; las features de ImageNet se reutilizan tal cual.
- **Umbral de decisión calibrado en 0,62, no 0,5.** Dejar pasar una uva podrida es
  más grave que descartar una sana, así que se exige más evidencia para decir
  "fresh". El umbral se barre sobre validación al entrenar y se guarda en
  `models/preprocessing.json`; si cambiás de dataset, se recalibra solo.
- **Split 70/15/15 estratificado y agrupado por imagen base.** Acá cada archivo es una
  imagen distinta, pero la lógica de agrupamiento se conserva porque protege el caso
  de un dataset con variantes de una misma foto — que es lo que pasaba con GrapeNet.
- **El preprocessing se serializa junto al modelo**, para que inferencia y
  entrenamiento apliquen exactamente la misma transformación.
- **Cambiar de dataset es una línea**: `config.DATASET_DIR` apunta a cualquier carpeta
  con subcarpetas `fresh/` y `rotten/`.

## Limitación conocida

El modelo funciona en el dominio con el que se entrenó: uvas verdes sobre superficies
claras y lisas, fotografiadas de cerca. Con racimos de uva negra en viñedo, con hojas
y fondos complejos, el rendimiento cae — la sección 6 del informe lo documenta con
números. Cerrar esa brecha requiere datos de ese tipo, no ajustes de modelo.

## Estado

- [x] Conseguir y validar el dataset
- [x] Implementar preprocessing y data augmentation
- [x] Definir y entrenar el modelo
- [x] Calibrar el umbral según el costo asimétrico del error
- [x] Evaluar y probar con imágenes nuevas
- [x] Guardar modelo y artefactos de preprocessing
- [x] Escribir el informe (`reports/report.md`)
