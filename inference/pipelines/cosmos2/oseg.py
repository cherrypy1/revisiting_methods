"""Cosmos-Predict2-2B, oseg (factory)."""
from ._patch import load_pipeline


def pipeline(device):
    return load_pipeline("oseg", device)
