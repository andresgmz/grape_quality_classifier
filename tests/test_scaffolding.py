"""Minimal scaffolding tests: TODO extend as the code gets implemented."""

from src import config


def test_directories_exist():
    assert config.RAW_DIR.exists()
    assert config.MODELS_DIR.exists()


def test_class_mapping():
    assert config.CLASSES == {"rotten": 0, "fresh": 1}


def test_fruit_is_grape():
    assert config.FRUIT == "grape"
