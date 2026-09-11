"""Tests del pipeline. No entrenan el modelo: solo la logica de datos."""

import pandas as pd
import pytest

from src import config, dataset
from src.prepare_data import base_image_id


def test_directories_exist():
    assert config.RAW_DIR.exists()
    assert config.MODELS_DIR.exists()


def test_class_mapping():
    assert config.CLASSES == {"rotten": 0, "fresh": 1}


def test_fruit_is_grape():
    assert config.FRUIT == "grape"


def test_base_image_id_quita_el_sufijo_de_aumentacion():
    assert (
        base_image_id("black_round_fresh_IMG_20250412_115529_115529_gamma_iter3.jpg")
        == "black_round_fresh_IMG_20250412_115529_115529"
    )


def test_base_image_id_agrupa_todas_las_variantes_de_una_foto():
    variantes = [
        "foto_IMG_1_brightness_contrast_iter1.jpg",
        "foto_IMG_1_gamma_iter5.jpg",
        "foto_IMG_1_horizontal_flip_iter2.jpg",
    ]
    assert len({base_image_id(v) for v in variantes}) == 1


def test_base_image_id_deja_intacto_un_nombre_sin_sufijo():
    assert base_image_id("foto_nueva.jpg") == "foto_nueva"


def _frame_de_prueba(n_fotos=20, variantes=5):
    filas = []
    for label in (0, 1):
        for foto in range(n_fotos):
            for it in range(1, variantes + 1):
                base = f"clase{label}_IMG_{foto}"
                filas.append(
                    {
                        "image": f"{base}_gamma_iter{it}.jpg",
                        "label": label,
                        "base_id": base,
                    }
                )
    return pd.DataFrame(filas)


def test_el_split_no_reparte_una_misma_foto_entre_dos_conjuntos():
    """El punto critico: sin esto habria fuga de datos y la precision seria falsa."""
    frame = dataset.split_train_val_test(_frame_de_prueba())
    splits_por_foto = frame.groupby("base_id")["split"].nunique()
    assert (splits_por_foto == 1).all()


def test_el_split_cubre_los_tres_conjuntos_con_las_dos_clases():
    frame = dataset.split_train_val_test(_frame_de_prueba())
    for split in ("train", "val", "test"):
        assert set(frame.loc[frame["split"] == split, "label"]) == {0, 1}


def test_subsample_conserva_todas_las_fotos_y_recorta_variantes():
    frame = _frame_de_prueba(n_fotos=10, variantes=5)
    reducido = dataset.subsample_variants(frame, per_photo=2)
    assert reducido["base_id"].nunique() == frame["base_id"].nunique()
    assert (reducido.groupby("base_id").size() == 2).all()


@pytest.mark.skipif(not config.LABELS_CSV.exists(), reason="labels.csv aun no generado")
def test_labels_csv_real_no_tiene_fuga_entre_splits():
    frame = dataset.load_labels()
    assert (frame.groupby("base_id")["split"].nunique() == 1).all()
    assert set(frame["label"]) == {0, 1}
