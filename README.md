# Grape Quality Classifier

CNN image classifier that detects fresh vs. rotten grapes.

Course project for **Advanced Topics in Software Engineering** (Master's, semester 1).
Fruit chosen by the group: **grape**. Binary task: `0 = rotten`, `1 = fresh`.

This repository currently contains the initial scaffolding only — module structure,
configuration and placeholders; the model is not implemented yet.

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

## Intended usage

```bash
pip install -r requirements.txt
python -m src.prepare_data       # extract the black grape subset into data/raw/
python -m src.dataset            # build labels.csv and splits
python -m src.train              # train and save the model
python -m src.evaluate           # metrics and confusion matrix
python -m src.predict <image_path>
```

## TODO

- [ ] Extract the GrapeNet black grape subset (see `data/README.md`)
- [ ] Implement preprocessing and data augmentation
- [ ] Define and train the CNN
- [ ] Evaluate and test with new images
- [ ] Save model and preprocessing artifacts
- [ ] Write the report (`reports/report.md`)
