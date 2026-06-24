"""Cosmos-Predict2-2B, pag (factory)."""
from ._patch import load_pipeline


def pipeline(device):
    return load_pipeline("pag", device)
