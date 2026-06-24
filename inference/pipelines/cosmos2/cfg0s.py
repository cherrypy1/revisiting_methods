"""Cosmos-Predict2-2B, cfg0s (factory)."""
from ._patch import load_pipeline


def pipeline(device):
    return load_pipeline("cfg0s", device)
