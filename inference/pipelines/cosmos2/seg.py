"""Cosmos-Predict2-2B, seg (factory)."""
from ._patch import load_pipeline


def pipeline(device):
    return load_pipeline("seg", device)
