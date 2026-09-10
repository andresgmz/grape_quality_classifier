"""Binary classification CNN (fresh vs. rotten grapes)."""

from __future__ import annotations

from . import config


def build_model(input_shape=None, num_classes: int = 2):
    """Simple CNN: stacked Conv2D + MaxPooling with a dense head. TODO."""
    raise NotImplementedError


def compile_model(model):
    """Compile with optimizer, binary loss and metrics. TODO: implement."""
    raise NotImplementedError
